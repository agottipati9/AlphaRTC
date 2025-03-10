/*
 *  Copyright (c) 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_REMOTE_BITRATE_ESTIMATOR_REMOTE_ESTIMATOR_PROXY_H_
#define MODULES_REMOTE_BITRATE_ESTIMATOR_REMOTE_ESTIMATOR_PROXY_H_

#include <map>
#include <vector>
#include <limits>
#include <queue>

#include "api/transport/network_control.h"
#include "api/transport/webrtc_key_value_config.h"
#include "modules/remote_bitrate_estimator/include/remote_bitrate_estimator.h"
#include "modules/third_party/onnxinfer/ONNXInferInterface.h"
#include "rtc_base/critical_section.h"
#include "rtc_base/experiments/field_trial_parser.h"
#include "rtc_base/numerics/sequence_number_util.h"
#include "modules/third_party/statcollect/StatCollect.h"

namespace webrtc {

class Clock;
class PacketRouter;
namespace rtcp {
class TransportFeedback;
}

// For tracking new features
// Packet information structure to store in the queue
struct PacketInfo {
    // Basic packet data
    int64_t arrival_time_ms;
    uint32_t send_time_ms;
    size_t payload_size;
    uint16_t sequence_number;
    uint32_t ssrc;
    uint8_t payload_type;
    
    // Derived or additional fields
    bool is_video;
    bool is_audio;
    bool is_probing;
    int transport_seq_num;  // Unwrapped transport sequence number
  };

// Class used when send-side BWE is enabled: This proxy is instantiated on the
// receive side. It buffers a number of receive timestamps and then sends
// transport feedback messages back too the send side.

class RemoteEstimatorProxy : public RemoteBitrateEstimator {
 public:
  RemoteEstimatorProxy(Clock* clock,
                       TransportFeedbackSenderInterface* feedback_sender,
                       const WebRtcKeyValueConfig* key_value_config,
                       NetworkStateEstimator* network_state_estimator);
  ~RemoteEstimatorProxy() override;

  void IncomingPacket(int64_t arrival_time_ms,
                      size_t payload_size,
                      const RTPHeader& header) override;
  void RemoveStream(uint32_t ssrc) override {}
  bool LatestEstimate(std::vector<unsigned int>* ssrcs,
                      unsigned int* bitrate_bps) const override;
  void OnRttUpdate(int64_t avg_rtt_ms, int64_t max_rtt_ms) override {}
  void SetMinBitrate(int min_bitrate_bps) override {}
  int64_t TimeUntilNextProcess() override;
  void Process() override;
  void OnBitrateChanged(int bitrate);
  void SetSendPeriodicFeedback(bool send_periodic_feedback);

 private:
  struct TransportWideFeedbackConfig {
    FieldTrialParameter<TimeDelta> back_window{"wind", TimeDelta::Millis(500)};
    FieldTrialParameter<TimeDelta> min_interval{"min", TimeDelta::Millis(50)};
    FieldTrialParameter<TimeDelta> max_interval{"max", TimeDelta::Millis(250)};
    FieldTrialParameter<TimeDelta> default_interval{"def",
                                                    TimeDelta::Millis(100)};
    FieldTrialParameter<double> bandwidth_fraction{"frac", 0.05};
    explicit TransportWideFeedbackConfig(
        const WebRtcKeyValueConfig* key_value_config) {
      ParseFieldTrial({&back_window, &min_interval, &max_interval,
                       &default_interval, &bandwidth_fraction},
                      key_value_config->Lookup(
                          "WebRTC-Bwe-TransportWideFeedbackIntervals"));
    }
  };

  static const int kMaxNumberOfPackets;
  void OnPacketArrival(uint16_t sequence_number,
                       int64_t arrival_time,
                       absl::optional<FeedbackRequest> feedback_request)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);
  void SendPeriodicFeedbacks() RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);
  void SendFeedbackOnRequest(int64_t sequence_number,
                             const FeedbackRequest& feedback_request)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);

  void SendbackBweEstimation(const BweMessage& bwe_message)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);
  bool TimeToSendBweMessage() RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);

  int64_t BuildFeedbackPacket(
      uint8_t feedback_packet_count,
      uint32_t media_ssrc,
      int64_t base_sequence_number,
      std::map<int64_t, int64_t>::const_iterator
          begin_iterator,  // |begin_iterator| is inclusive.
      std::map<int64_t, int64_t>::const_iterator
          end_iterator,  // |end_iterator| is exclusive.
      rtcp::TransportFeedback* feedback_packet);

  uint32_t GetTtimeFromAbsSendtime(uint32_t absoluteSendTime)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(&lock_);

  Clock* const clock_;
  TransportFeedbackSenderInterface* const feedback_sender_;
  const TransportWideFeedbackConfig send_config_;
  int64_t last_process_time_ms_;

  rtc::CriticalSection lock_;
  //  |network_state_estimator_| may be null.
  NetworkStateEstimator* const network_state_estimator_
      RTC_PT_GUARDED_BY(&lock_);
  uint32_t media_ssrc_ RTC_GUARDED_BY(&lock_);
  uint8_t feedback_packet_count_ RTC_GUARDED_BY(&lock_);
  SeqNumUnwrapper<uint16_t> unwrapper_ RTC_GUARDED_BY(&lock_);
  absl::optional<int64_t> periodic_window_start_seq_ RTC_GUARDED_BY(&lock_);
  // Map unwrapped seq -> time.
  std::map<int64_t, int64_t> packet_arrival_times_ RTC_GUARDED_BY(&lock_);
  int64_t send_interval_ms_ RTC_GUARDED_BY(&lock_);
  bool send_periodic_feedback_ RTC_GUARDED_BY(&lock_);

  // Bandwidth estimation sending back
  int64_t bwe_sendback_interval_ms_ RTC_GUARDED_BY(&lock_);
  int64_t last_bwe_sendback_ms_ RTC_GUARDED_BY(&lock_);

  // StatCollect moudule
  StatCollect::StatsCollectModule stats_collect_;
  int cycles_ RTC_GUARDED_BY(&lock_);
  uint32_t max_abs_send_time_ RTC_GUARDED_BY(&lock_);
  void* onnx_infer_;

// ********* for tracking new features *********
std::queue<PacketInfo> packet_queue_;

 // Handle clock offset
 int64_t time_offset_ = -1;
 int64_t expected_min_delay_ms_ = 40; // Default value ie. 10ms
  
  // Measurement interval tracking
  int64_t last_metrics_calculation_ms_ = -1;
  int64_t measurement_interval_ms_ = 60;  // Default 60ms
  int64_t last_feedback_report_ms_ = 0;
  

  // Metrics state
  int64_t min_delay_ms_overall_ = 1000;

  // Methods for metrics calculation
  void ProcessMetricsInterval();
  bool IsTimeForMetricsCalculation(int64_t now_ms) const;

    // Helper function to determine packet type
bool IsVideoPacket(uint8_t payload_type) const;
bool IsAudioPacket(uint8_t payload_type) const;
bool IsProbingPacket(uint8_t payload_type) const;

// Storage for computed metrics (vector-based)
struct MIMetrics {  
  const size_t DEFAULT_HISTORY_SIZE = 10;
  
  // Basic metrics
  std::vector<int64_t> receiving_rate_bps = std::vector<int64_t>(DEFAULT_HISTORY_SIZE, 0);
  std::vector<int> received_packets = std::vector<int>(DEFAULT_HISTORY_SIZE, 0);
  std::vector<size_t> received_bytes = std::vector<size_t>(DEFAULT_HISTORY_SIZE, 0);
  
  // Delay metrics (OWD)
  std::vector<double> queuing_delay_ms = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
  std::vector<double> delay_ms = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
  std::vector<int64_t> minimum_seen_delay_ms = std::vector<int64_t>(DEFAULT_HISTORY_SIZE, 1000);
  std::vector<double> delay_ratio = std::vector<double>(DEFAULT_HISTORY_SIZE, 1.0);
  std::vector<double> delay_avg_min_difference_ms = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
  
  // Packet timing metrics
  std::vector<double> packet_interarrival_time_ms = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
  std::vector<double> packet_jitter_ms = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
  
  // Loss metrics
  std::vector<double> packet_loss_ratio = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
  std::vector<int> average_lost_packets = std::vector<int>(DEFAULT_HISTORY_SIZE, 0);
  
  // Packet type metrics
  std::vector<double> video_packets_probability = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
  std::vector<double> audio_packets_probability = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
  std::vector<double> probing_packets_probability = std::vector<double>(DEFAULT_HISTORY_SIZE, 0.0);
      
  // Feedback metrics
  std::vector<int64_t> timesteps_since_last_feedback_ms = std::vector<int64_t>(DEFAULT_HISTORY_SIZE, 0);

  // Action metrics
  std::vector<int> previous_actions_ = std::vector<int>(DEFAULT_HISTORY_SIZE, 0);
  
  // Helper method to update a metric (adds value to end, pops front)
  template<typename T>
  void updateMetric(std::vector<T>& metric, const T& value) {
    metric.push_back(value);
    metric.erase(metric.begin());
  }

  // Helper method to convert a vector to a string representation
template<typename T>
std::string vectorToString(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        ss << vec[i];
        if (i < vec.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}
};
  MIMetrics mi_metrics_;  

// ********* for tracking new features *********


};

}  // namespace webrtc

#endif  //  MODULES_REMOTE_BITRATE_ESTIMATOR_REMOTE_ESTIMATOR_PROXY_H_
