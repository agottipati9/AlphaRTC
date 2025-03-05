/*
 *  Copyright (c) 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "modules/remote_bitrate_estimator/remote_estimator_proxy.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <algorithm>

#include "api/alphacc_config.h"
#include "modules/rtp_rtcp/source/rtcp_packet/transport_feedback.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_minmax.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

// Impossible to request feedback older than what can be represented by 15 bits.
const int RemoteEstimatorProxy::kMaxNumberOfPackets = (1 << 15);

// The maximum allowed value for a timestamp in milliseconds. This is lower
// than the numerical limit since we often convert to microseconds.
static constexpr int64_t kMaxTimeMs =
    std::numeric_limits<int64_t>::max() / 1000;

RemoteEstimatorProxy::RemoteEstimatorProxy(
    Clock* clock,
    TransportFeedbackSenderInterface* feedback_sender,
    const WebRtcKeyValueConfig* key_value_config,
    NetworkStateEstimator* network_state_estimator)
    : clock_(clock),
      feedback_sender_(feedback_sender),
      send_config_(key_value_config),
      last_process_time_ms_(-1),
      network_state_estimator_(network_state_estimator),
      media_ssrc_(0),
      feedback_packet_count_(0),
      send_interval_ms_(send_config_.default_interval->ms()),
      send_periodic_feedback_(true),
      bwe_sendback_interval_ms_(GetAlphaCCConfig()->bwe_feedback_duration_ms),
      last_bwe_sendback_ms_(clock->TimeInMilliseconds()),
      stats_collect_(StatCollect::SC_TYPE_STRUCT),
      cycles_(-1),
      max_abs_send_time_(0) {
      // previous_abs_send_time_(0),
      // abs_send_timestamp_(clock->CurrentTime()) {
  RTC_LOG(LS_INFO)
      << "Maximum interval between transport feedback RTCP messages (ms): "
      << send_config_.max_interval->ms();
}

RemoteEstimatorProxy::~RemoteEstimatorProxy() {}

void RemoteEstimatorProxy::IncomingPacket(int64_t arrival_time_ms,
                                          size_t payload_size,
                                          const RTPHeader& header) {
  if (arrival_time_ms < 0 || arrival_time_ms > kMaxTimeMs) {
    RTC_LOG(LS_WARNING) << "Arrival time out of bounds: " << arrival_time_ms;
    return;
  }
  rtc::CritScope cs(&lock_);
  media_ssrc_ = header.ssrc;
  int64_t seq = 0;

  uint32_t send_time_ms =
      GetTtimeFromAbsSendtime(header.extension.absoluteSendTime);
  bool time_to_send_bew_message = TimeToSendBweMessage();
  float estimation = 0;

  // Added for tracking new features
  // Handle clock offset
  if (time_offset_ == -1) {
    time_offset_ = arrival_time_ms - send_time_ms - expected_min_delay_ms_;
  }

  // Create packet info structure
  PacketInfo packet_info;
  packet_info.arrival_time_ms = arrival_time_ms - time_offset_;
  packet_info.send_time_ms = send_time_ms;
  packet_info.payload_size = payload_size;
  packet_info.sequence_number = header.sequenceNumber;
  packet_info.ssrc = header.ssrc;
  packet_info.payload_type = header.payloadType;

  // Determine packet type
  packet_info.is_video = IsVideoPacket(header.payloadType);
  packet_info.is_audio = IsAudioPacket(header.payloadType);
  packet_info.is_probing = IsProbingPacket(header.payloadType);

  // Add transport sequence number if available
  if (header.extension.hasTransportSequenceNumber) {
    packet_info.transport_seq_num = 
        unwrapper_.Unwrap(header.extension.transportSequenceNumber);
  } else {
    packet_info.transport_seq_num = -1;  // Not available
  }
  
  // Add to queue
  packet_queue_.push(packet_info);
  
  if (header.extension.hasTransportSequenceNumber) {
    seq = unwrapper_.Unwrap(header.extension.transportSequenceNumber);

    if (send_periodic_feedback_) {
      if (periodic_window_start_seq_ &&
          packet_arrival_times_.lower_bound(*periodic_window_start_seq_) ==
              packet_arrival_times_.end()) {
        // Start new feedback packet, cull old packets.
        for (auto it = packet_arrival_times_.begin();
             it != packet_arrival_times_.end() && it->first < seq &&
             arrival_time_ms - it->second >= send_config_.back_window->ms();) {
          it = packet_arrival_times_.erase(it);
        }
      }
      if (!periodic_window_start_seq_ || seq < *periodic_window_start_seq_) {
        periodic_window_start_seq_ = seq;
      }
    }

    // We are only interested in the first time a packet is received.
    if (packet_arrival_times_.find(seq) != packet_arrival_times_.end())
      return;

    packet_arrival_times_[seq] = arrival_time_ms;

    // Limit the range of sequence numbers to send feedback for.
    auto first_arrival_time_to_keep = packet_arrival_times_.lower_bound(
        packet_arrival_times_.rbegin()->first - kMaxNumberOfPackets);
    if (first_arrival_time_to_keep != packet_arrival_times_.begin()) {
      packet_arrival_times_.erase(packet_arrival_times_.begin(),
                                  first_arrival_time_to_keep);
      if (send_periodic_feedback_) {
        // |packet_arrival_times_| cannot be empty since we just added one
        // element and the last element is not deleted.
        RTC_DCHECK(!packet_arrival_times_.empty());
        periodic_window_start_seq_ = packet_arrival_times_.begin()->first;
      }
    }

    if (header.extension.feedback_request) {
      // Send feedback packet immediately.
      SendFeedbackOnRequest(seq, header.extension.feedback_request.value());
    }
  }

  // Save per-packet info locally on receiving
  // ---------- Collect packet-related info into a local file ----------
  double pacing_rate =
      time_to_send_bew_message ? estimation : SC_PACER_PACING_RATE_EMPTY;
  double padding_rate =
      time_to_send_bew_message ? estimation : SC_PACER_PADDING_RATE_EMPTY;

  // Save per-packet info locally on receiving
  auto res = stats_collect_.StatsCollect(
      pacing_rate, padding_rate, header.payloadType,
                              header.sequenceNumber, send_time_ms, header.ssrc,
                              header.paddingLength, header.headerLength,
                              arrival_time_ms, payload_size, 0);
  if (res != StatCollect::SCResult::SC_SUCCESS)
  {
    RTC_LOG(LS_ERROR) << "Collect data failed";
  }
  std::string out_data = stats_collect_.DumpData();
  if (out_data.empty())
  {
    RTC_LOG(LS_ERROR) << "Save data failed";
  }

  RTC_LOG(LS_INFO) << out_data;

  // if (network_state_estimator_ && header.extension.hasAbsoluteSendTime) {
  //   PacketResult packet_result;
  //   packet_result.receive_time = Timestamp::Millis(arrival_time_ms);
  //   // Ignore reordering of packets and assume they have approximately the same
  //   // send time.
  //   abs_send_timestamp_ += std::max(
  //       header.extension.GetAbsoluteSendTimeDelta(previous_abs_send_time_),
  //       TimeDelta::Millis(0));
  //   previous_abs_send_time_ = header.extension.absoluteSendTime;
  //   packet_result.sent_packet.send_time = abs_send_timestamp_;
  //   // TODO(webrtc:10742): Take IP header and transport overhead into account.
  //   packet_result.sent_packet.size =
  //       DataSize::Bytes(header.headerLength + payload_size);
  //   packet_result.sent_packet.sequence_number = seq;
  //   network_state_estimator_->OnReceivedPacket(packet_result);
  // }
}

bool RemoteEstimatorProxy::LatestEstimate(std::vector<unsigned int>* ssrcs,
                                          unsigned int* bitrate_bps) const {
  return false;
}

int64_t RemoteEstimatorProxy::TimeUntilNextProcess() {
  rtc::CritScope cs(&lock_);
  if (!send_periodic_feedback_) {
    // Wait a day until next process.
    return 24 * 60 * 60 * 1000;
  } else if (last_process_time_ms_ != -1) {
    int64_t now = clock_->TimeInMilliseconds();
    if (now - last_process_time_ms_ < send_interval_ms_)
      return last_process_time_ms_ + send_interval_ms_ - now;
  }
  return 0;
}

void RemoteEstimatorProxy::Process() {
  rtc::CritScope cs(&lock_);
  if (!send_periodic_feedback_) {
    return;
  }
  last_process_time_ms_ = clock_->TimeInMilliseconds();

  // ******* Added for tracking new features ********
  // Check if we need to process packet queue
  if (IsTimeForMetricsCalculation(last_process_time_ms_)) {
    ProcessMetricsInterval();
  }

  SendPeriodicFeedbacks();
}

void RemoteEstimatorProxy::OnBitrateChanged(int bitrate_bps) {
  // TwccReportSize = Ipv4(20B) + UDP(8B) + SRTP(10B) +
  // AverageTwccReport(30B)
  // TwccReport size at 50ms interval is 24 byte.
  // TwccReport size at 250ms interval is 36 byte.
  // AverageTwccReport = (TwccReport(50ms) + TwccReport(250ms)) / 2
  constexpr int kTwccReportSize = 20 + 8 + 10 + 30;
  const double kMinTwccRate =
      kTwccReportSize * 8.0 * 1000.0 / send_config_.max_interval->ms();
  const double kMaxTwccRate =
      kTwccReportSize * 8.0 * 1000.0 / send_config_.min_interval->ms();

  // set previous actions -- GCC (if using ONNX and Pyinfer, this is not needed, we can get output directly)
  mi_metrics_.updateMetric(mi_metrics_.previous_actions_, bitrate_bps);

  // Let TWCC reports occupy 5% of total bandwidth.
  rtc::CritScope cs(&lock_);
  send_interval_ms_ = static_cast<int>(
      0.5 + kTwccReportSize * 8.0 * 1000.0 /
                rtc::SafeClamp(send_config_.bandwidth_fraction * bitrate_bps,
                               kMinTwccRate, kMaxTwccRate));
}

void RemoteEstimatorProxy::SetSendPeriodicFeedback(
    bool send_periodic_feedback) {
  rtc::CritScope cs(&lock_);
  send_periodic_feedback_ = send_periodic_feedback;
}


void RemoteEstimatorProxy::OnPacketArrival(
    uint16_t sequence_number,
    int64_t arrival_time,
    absl::optional<FeedbackRequest> feedback_request) {
  if (arrival_time < 0 || arrival_time > kMaxTimeMs) {
    RTC_LOG(LS_WARNING) << "Arrival time out of bounds: " << arrival_time;
    return;
  }

  int64_t seq = unwrapper_.Unwrap(sequence_number);

  if (send_periodic_feedback_) {
    if (periodic_window_start_seq_ &&
        packet_arrival_times_.lower_bound(*periodic_window_start_seq_) ==
            packet_arrival_times_.end()) {
      // Start new feedback packet, cull old packets.
      for (auto it = packet_arrival_times_.begin();
           it != packet_arrival_times_.end() && it->first < seq &&
           arrival_time - it->second >= send_config_.back_window->ms();) {
        it = packet_arrival_times_.erase(it);
      }
    }
    if (!periodic_window_start_seq_ || seq < *periodic_window_start_seq_) {
      periodic_window_start_seq_ = seq;
    }
  }

  // We are only interested in the first time a packet is received.
  if (packet_arrival_times_.find(seq) != packet_arrival_times_.end())
    return;

  packet_arrival_times_[seq] = arrival_time;

  // Limit the range of sequence numbers to send feedback for.
  auto first_arrival_time_to_keep = packet_arrival_times_.lower_bound(
      packet_arrival_times_.rbegin()->first - kMaxNumberOfPackets);
  if (first_arrival_time_to_keep != packet_arrival_times_.begin()) {
    packet_arrival_times_.erase(packet_arrival_times_.begin(),
                                first_arrival_time_to_keep);
    if (send_periodic_feedback_) {
      // |packet_arrival_times_| cannot be empty since we just added one element
      // and the last element is not deleted.
      RTC_DCHECK(!packet_arrival_times_.empty());
      periodic_window_start_seq_ = packet_arrival_times_.begin()->first;
    }
  }

  if (feedback_request) {
    // Send feedback packet immediately.
    SendFeedbackOnRequest(seq, *feedback_request);
  }
}

bool RemoteEstimatorProxy::TimeToSendBweMessage() {
  int64_t time_now = clock_->TimeInMilliseconds();
  if (time_now - bwe_sendback_interval_ms_ > last_bwe_sendback_ms_) {
    last_bwe_sendback_ms_ = time_now;
    return true;
  }
  return false;
}

void RemoteEstimatorProxy::SendPeriodicFeedbacks() {
  // |periodic_window_start_seq_| is the first sequence number to include in the
  // current feedback packet. Some older may still be in the map, in case a
  // reordering happens and we need to retransmit them.
  if (!periodic_window_start_seq_)
    return;

  std::unique_ptr<rtcp::RemoteEstimate> remote_estimate;
  if (network_state_estimator_) {
    absl::optional<NetworkStateEstimate> state_estimate =
        network_state_estimator_->GetCurrentEstimate();
    if (state_estimate) {
      remote_estimate = std::make_unique<rtcp::RemoteEstimate>();
      remote_estimate->SetEstimate(state_estimate.value());
    }
  }

  for (auto begin_iterator =
           packet_arrival_times_.lower_bound(*periodic_window_start_seq_);
       begin_iterator != packet_arrival_times_.cend();
       begin_iterator =
           packet_arrival_times_.lower_bound(*periodic_window_start_seq_)) {
    auto feedback_packet = std::make_unique<rtcp::TransportFeedback>();
    periodic_window_start_seq_ = BuildFeedbackPacket(
        feedback_packet_count_++, media_ssrc_, *periodic_window_start_seq_,
        begin_iterator, packet_arrival_times_.cend(), feedback_packet.get());

    RTC_DCHECK(feedback_sender_ != nullptr);

    std::vector<std::unique_ptr<rtcp::RtcpPacket>> packets;
    if (remote_estimate) {
      packets.push_back(std::move(remote_estimate));
    }
    packets.push_back(std::move(feedback_packet));

    // ****** Added for tracking new features ******
    last_feedback_report_ms_ = clock_->TimeInMilliseconds();

    feedback_sender_->SendCombinedRtcpPacket(std::move(packets));
    // Note: Don't erase items from packet_arrival_times_ after sending, in case
    // they need to be re-sent after a reordering. Removal will be handled
    // by OnPacketArrival once packets are too old.
  }
}

void RemoteEstimatorProxy::SendFeedbackOnRequest(
    int64_t sequence_number,
    const FeedbackRequest& feedback_request) {
  if (feedback_request.sequence_count == 0) {
    return;
  }

  auto feedback_packet = std::make_unique<rtcp::TransportFeedback>(
      feedback_request.include_timestamps);

  int64_t first_sequence_number =
      sequence_number - feedback_request.sequence_count + 1;
  auto begin_iterator =
      packet_arrival_times_.lower_bound(first_sequence_number);
  auto end_iterator = packet_arrival_times_.upper_bound(sequence_number);

  BuildFeedbackPacket(feedback_packet_count_++, media_ssrc_,
                      first_sequence_number, begin_iterator, end_iterator,
                      feedback_packet.get());

  // Clear up to the first packet that is included in this feedback packet.
  packet_arrival_times_.erase(packet_arrival_times_.begin(), begin_iterator);

  RTC_DCHECK(feedback_sender_ != nullptr);
  std::vector<std::unique_ptr<rtcp::RtcpPacket>> packets;
  packets.push_back(std::move(feedback_packet));

  // ****** Added for tracking new features ******
  last_feedback_report_ms_ = clock_->TimeInMilliseconds();

  feedback_sender_->SendCombinedRtcpPacket(std::move(packets));
}

void RemoteEstimatorProxy::SendbackBweEstimation(const BweMessage& bwe) {
  auto app_packet = std::make_unique<rtcp::App>();
  app_packet->SetSubType(kAppPacketSubType);
  app_packet->SetName(kAppPacketName);

  app_packet->SetData(reinterpret_cast<const uint8_t*>(&bwe), sizeof(bwe));
  std::vector<std::unique_ptr<rtcp::RtcpPacket>> packets;
  packets.push_back(std::move(app_packet));
  feedback_sender_->SendCombinedRtcpPacket(std::move(packets));
}

int64_t RemoteEstimatorProxy::BuildFeedbackPacket(
    uint8_t feedback_packet_count,
    uint32_t media_ssrc,
    int64_t base_sequence_number,
    std::map<int64_t, int64_t>::const_iterator begin_iterator,
    std::map<int64_t, int64_t>::const_iterator end_iterator,
    rtcp::TransportFeedback* feedback_packet) {
  RTC_DCHECK(begin_iterator != end_iterator);

  // TODO(sprang): Measure receive times in microseconds and remove the
  // conversions below.
  feedback_packet->SetMediaSsrc(media_ssrc);
  // Base sequence number is the expected first sequence number. This is known,
  // but we might not have actually received it, so the base time shall be the
  // time of the first received packet in the feedback.
  feedback_packet->SetBase(static_cast<uint16_t>(base_sequence_number & 0xFFFF),
                           begin_iterator->second * 1000);
  feedback_packet->SetFeedbackSequenceNumber(feedback_packet_count);
  int64_t next_sequence_number = base_sequence_number;
  for (auto it = begin_iterator; it != end_iterator; ++it) {
    if (!feedback_packet->AddReceivedPacket(
            static_cast<uint16_t>(it->first & 0xFFFF), it->second * 1000)) {
      // If we can't even add the first seq to the feedback packet, we won't be
      // able to build it at all.
      RTC_CHECK(begin_iterator != it);

      // Could not add timestamp, feedback packet might be full. Return and
      // try again with a fresh packet.
      break;
    }
    next_sequence_number = it->first + 1;
  }
  return next_sequence_number;
}

uint32_t RemoteEstimatorProxy::GetTtimeFromAbsSendtime(
    uint32_t absoluteSendTime) {
  if (cycles_ == -1) {
    // Initalize
    max_abs_send_time_ = absoluteSendTime;
    cycles_ = 0;
  }
  // Abs sender time is 24 bit 6.18 fixed point. Shift by 8 to normalize to
  // 32 bits (unsigned). Calculate the difference between this packet's
  // send time and the maximum observed. Cast to signed 32-bit to get the
  // desired wrap-around behavior.
  if (static_cast<int32_t>((absoluteSendTime << 8) -
                           (max_abs_send_time_ << 8)) >= 0) {
    // The difference is non-negative, meaning that this packet is newer
    // than the previously observed maximum absolute send time.
    if (absoluteSendTime < max_abs_send_time_) {
      // Wrap detected.
      cycles_++;
    }
    max_abs_send_time_ = absoluteSendTime;
  }
  // Abs sender time is 24 bit 6.18 fixed point. Divide by 2^18 to convert
  // to floating point representation.
  double send_time_seconds =
      static_cast<double>(absoluteSendTime) / 262144 + 64.0 * cycles_;
  uint32_t send_time_ms =
      static_cast<uint32_t>(std::round(send_time_seconds * 1000));
  return send_time_ms;
}

// Added for tracking new features

bool RemoteEstimatorProxy::IsVideoPacket(uint8_t payload_type) const {
  // Define payload type ranges for video (adjust based on your codec configuration)
  return (payload_type >= 96 && payload_type <= 127);
}

bool RemoteEstimatorProxy::IsAudioPacket(uint8_t payload_type) const {
  // Define payload type ranges for audio (adjust based on your codec configuration)
  return (payload_type >= 0 && payload_type <= 95);
}

bool RemoteEstimatorProxy::IsProbingPacket(uint8_t payload_type) const {
  // Define payload type for probing packets if applicable
  return false; // Implement based on your probing mechanism
}

bool RemoteEstimatorProxy::IsTimeForMetricsCalculation(int64_t now_ms) const {
  return (last_metrics_calculation_ms_ == -1 || 
          now_ms - last_metrics_calculation_ms_ >= measurement_interval_ms_);
}

// Metrics calculation method that processes the queue
void RemoteEstimatorProxy::ProcessMetricsInterval() {
  int64_t now_ms = clock_->TimeInMilliseconds();
  last_metrics_calculation_ms_ = now_ms;
  
  if (packet_queue_.empty()) {
    // No packets to process
    return;
  }
  
  // Initialize counters and data structures for this interval
  size_t total_bytes = 0;
  int total_packets = 0;
  int video_packets = 0;
  int audio_packets = 0;
  int probing_packets = 0;
  
  // Delay tracking
  int64_t sum_delay_ms = 0;
  int64_t min_delay_ms_this_interval = std::numeric_limits<int64_t>::max();
  
  // For interarrival calculation
  std::vector<int64_t> arrival_times;
  
  // For loss detection
  std::map<int64_t, bool> received_seq_nums;
  int64_t min_seq = std::numeric_limits<int64_t>::max();
  int64_t max_seq = -1;
  
  // Process all packets in the queue
  while (!packet_queue_.empty()) {
    const PacketInfo& packet = packet_queue_.front();
    
    // Basic packet statistics
    total_bytes += packet.payload_size;
    total_packets++;
    
    // Packet type counting
    if (packet.is_video) video_packets++;
    if (packet.is_audio) audio_packets++;
    if (packet.is_probing) probing_packets++;
    
    // Store arrival time for interarrival calculation
    arrival_times.push_back(packet.arrival_time_ms);
    
    // Calculate packet delay
    int64_t packet_delay_ms = packet.arrival_time_ms - packet.send_time_ms;
    packet_delay_ms = std::min(packet_delay_ms, 1000L); // Cap at 1 second
    if (packet_delay_ms >= 0) {  // Ignore negative delays
      sum_delay_ms += packet_delay_ms;
      
      // Update minimum delays
      if (packet_delay_ms < min_delay_ms_overall_) {
        min_delay_ms_overall_ = packet_delay_ms;
      }
      if (packet_delay_ms < min_delay_ms_this_interval) {
        min_delay_ms_this_interval = packet_delay_ms;
      }
    }
    
    // Track sequence numbers for loss calculation
    if (packet.transport_seq_num != -1) {
      received_seq_nums[packet.transport_seq_num] = true;
      min_seq = std::min(min_seq, static_cast<int64_t>(packet.transport_seq_num));
      max_seq = std::max(max_seq, static_cast<int64_t>(packet.transport_seq_num));
    }
    
    // Remove the processed packet
    packet_queue_.pop();
  }
  
  // Calculate receiving rate
  double interval_duration_sec = measurement_interval_ms_ / 1000.0;
  int64_t receiving_rate_bps = static_cast<int64_t>(total_bytes * 8 / interval_duration_sec);
  
  // Calculate average delay
  double avg_delay_ms = (total_packets > 0) ? 
      static_cast<double>(sum_delay_ms) / total_packets : 0;
  
  // Calculate queuing delay
  double queuing_delay_ms = (total_packets > 0) ? 
      avg_delay_ms - min_delay_ms_overall_ : 0;
  
  // Calculate delay with fixed base
  // const int64_t fixed_base_delay_ms = 200;  // Configurable base delay
  // double delay_ms = (total_packets > 0) ? 
  //     avg_delay_ms - fixed_base_delay_ms : 0;
  
  // Calculate delay ratio
  double delay_ratio = (min_delay_ms_this_interval != std::numeric_limits<int64_t>::max() && min_delay_ms_this_interval > 0) ? 
      avg_delay_ms / min_delay_ms_this_interval : 1.0;
  
  // Calculate delay average/min difference
  double delay_avg_min_difference_ms = (min_delay_ms_this_interval != std::numeric_limits<int64_t>::max()) ? 
      avg_delay_ms - min_delay_ms_this_interval : 0;
  
  // Calculate interarrival metrics
  double mean_interarrival_ms = 0;
  double jitter_ms = 0;
  
  if (arrival_times.size() > 1) {
    std::sort(arrival_times.begin(), arrival_times.end());
    
    std::vector<double> interarrival_times;
    for (size_t i = 1; i < arrival_times.size(); i++) {
      interarrival_times.push_back(arrival_times[i] - arrival_times[i-1]);
    }
    
    // Calculate mean
    double sum = 0;
    for (const auto& time : interarrival_times) {
      sum += time;
    }
    mean_interarrival_ms = sum / interarrival_times.size();
    
    // Calculate standard deviation (jitter)
    double sq_sum = 0;
    for (const auto& time : interarrival_times) {
      sq_sum += (time - mean_interarrival_ms) * (time - mean_interarrival_ms);
    }
    jitter_ms = std::sqrt(sq_sum / interarrival_times.size());
  }
  
  // Calculate loss metrics
  int lost_packets = 0;
  double packet_loss_ratio = 0;
  int average_lost_packets = 0;
  
  if (min_seq != std::numeric_limits<int64_t>::max() && max_seq != -1) {
    // Calculate expected number of packets
    int expected_packets = max_seq - min_seq + 1;
    
    // Calculate lost packets
    lost_packets = expected_packets - received_seq_nums.size();
    
    // Calculate loss ratio
    packet_loss_ratio = (expected_packets > 0) ? 
        static_cast<double>(lost_packets) / expected_packets : 0;
        
    // For average lost packets, we'd need to track loss bursts
    // This is a simplified version
    average_lost_packets = lost_packets;
  }
  
  // Calculate packet type probabilities
  double video_packets_probability = (total_packets > 0) ? 
      static_cast<double>(video_packets) / total_packets : 0;
      
  double audio_packets_probability = (total_packets > 0) ? 
      static_cast<double>(audio_packets) / total_packets : 0;
      
  double probing_packets_probability = (total_packets > 0) ? 
      static_cast<double>(probing_packets) / total_packets : 0;
  
  // Store computed metrics
  // rate metrics
  mi_metrics_.updateMetric(mi_metrics_.receiving_rate_bps, receiving_rate_bps);
  mi_metrics_.updateMetric(mi_metrics_.received_packets, total_packets);
  mi_metrics_.updateMetric(mi_metrics_.received_bytes, total_bytes);
  RTC_LOG(LS_INFO) << "Receiving rate (bps): " << mi_metrics_.vectorToString(mi_metrics_.receiving_rate_bps);

  // delay metrics
  mi_metrics_.updateMetric(mi_metrics_.queuing_delay_ms, queuing_delay_ms);
  mi_metrics_.updateMetric(mi_metrics_.delay_ms, avg_delay_ms);
  mi_metrics_.updateMetric(mi_metrics_.minimum_seen_delay_ms, min_delay_ms_overall_);
  mi_metrics_.updateMetric(mi_metrics_.delay_ratio, delay_ratio);
  mi_metrics_.updateMetric(mi_metrics_.delay_avg_min_difference_ms, delay_avg_min_difference_ms);
  RTC_LOG(LS_INFO) << "Queuing delay (ms): " << mi_metrics_.vectorToString(mi_metrics_.queuing_delay_ms);
  RTC_LOG(LS_INFO) << "One Way Delay (ms): " << mi_metrics_.vectorToString(mi_metrics_.delay_ms);
  RTC_LOG(LS_INFO) << "Minimum seen OWD delay (ms): " << mi_metrics_.vectorToString(mi_metrics_.minimum_seen_delay_ms);
  RTC_LOG(LS_INFO) << "Delay ratio: " << mi_metrics_.vectorToString(mi_metrics_.delay_ratio);
  RTC_LOG(LS_INFO) << "Delay average min difference (ms): " << mi_metrics_.vectorToString(mi_metrics_.delay_avg_min_difference_ms);

  // jitter metrics
  mi_metrics_.updateMetric(mi_metrics_.packet_interarrival_time_ms, mean_interarrival_ms);
  mi_metrics_.updateMetric(mi_metrics_.packet_jitter_ms, jitter_ms);
  RTC_LOG(LS_INFO) << "Packet interarrival time (ms): " << mi_metrics_.vectorToString(mi_metrics_.packet_interarrival_time_ms);
  RTC_LOG(LS_INFO) << "Packet jitter (ms): " << mi_metrics_.vectorToString(mi_metrics_.packet_jitter_ms);
 
  // packet loss metrics
  mi_metrics_.updateMetric(mi_metrics_.packet_loss_ratio, packet_loss_ratio);
  mi_metrics_.updateMetric(mi_metrics_.average_lost_packets, average_lost_packets);
  RTC_LOG(LS_INFO) << "Packet loss ratio: " << mi_metrics_.vectorToString(mi_metrics_.packet_loss_ratio);
  RTC_LOG(LS_INFO) << "Average lost packets: " << mi_metrics_.vectorToString(mi_metrics_.average_lost_packets);

  // packet type metrics
  mi_metrics_.updateMetric(mi_metrics_.video_packets_probability, video_packets_probability);
  mi_metrics_.updateMetric(mi_metrics_.audio_packets_probability, audio_packets_probability);
  mi_metrics_.updateMetric(mi_metrics_.probing_packets_probability, probing_packets_probability);
  RTC_LOG(LS_INFO) << "Video packets probability: " << mi_metrics_.vectorToString(mi_metrics_.video_packets_probability);
  RTC_LOG(LS_INFO) << "Audio packets probability: " << mi_metrics_.vectorToString(mi_metrics_.audio_packets_probability);
  
  // misc metrics
  mi_metrics_.updateMetric(mi_metrics_.timesteps_since_last_feedback_ms, static_cast<int64_t>((now_ms - last_feedback_report_ms_) / measurement_interval_ms_));
  RTC_LOG(LS_INFO) << "Timesteps since last feedback: " << mi_metrics_.vectorToString(mi_metrics_.timesteps_since_last_feedback_ms);

  // previous actions
  RTC_LOG(LS_INFO) << "Previous actions: " << mi_metrics_.vectorToString(mi_metrics_.previous_actions_);
}

}  // namespace webrtc
