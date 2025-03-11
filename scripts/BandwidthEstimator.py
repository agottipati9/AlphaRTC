import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F

import os


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=256, init_w=3e-3, log_std_min=-10, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # Note: actor outputs are [0, 1]
        x = (x + 1) / 2
        return x
    
    def evaluate(self, state, epsilon=1e-6):
        mu = self.forward(state)
        return mu
        
    def get_action(self, state):
        mu = self.forward(state)
        dist = torch.distributions.Normal(mu, 1)
        action = dist.sample()
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu = self.forward(state)
        return mu.detach().cpu()
    

class PacketInfo:
    def __init__(self, payload_size=0, arrival_time_ms=0, send_time_ms=0, transport_seq_num=-1):
        self.payload_size = payload_size
        self.arrival_time_ms = arrival_time_ms
        self.send_time_ms = send_time_ms
        self.transport_seq_num = transport_seq_num
        self.is_video = False
        self.is_audio = False
        self.is_probing = False

class Estimator(object):
    def __init__(self):
        # Constants
        self.max_bwe = int(8e6)  # 8 Mbps
        self.log_max_mbps = np.log(self.max_bwe / 1e6)
        self.min_bwe = int(1e4)  # 10 Kbps
        self.log_min_mbps = np.log(self.min_bwe / 1e6)
        self.history_window_size = 10
        self.max_delay_ms = 1000
        self.max_lost_packets = 100
        self.measurement_interval_ms = 60
        # Bandwidth Estimation
        self.bwe = self.min_bwe
        # Packet Queue
        self.packet_queue = []
        # Receiving Rate metrics
        self.receiving_rate_history = np.zeros(self.history_window_size)
        # OWD Delay Metrics
        self.queuing_delay_history = np.zeros(self.history_window_size)
        self.delay_history = np.zeros(self.history_window_size)
        self.min_seen_delay_history = np.ones(self.history_window_size) * 1000
        self.delay_ratio_history = np.ones(self.history_window_size)
        self.delay_avg_min_difference_history = np.zeros(self.history_window_size)
        self.packet_loss_history = np.zeros(self.history_window_size)
        self.min_delay_ms_overall = 1000
        # Packet Timing Metrics
        self.packet_interarrival_time_history = np.zeros(self.history_window_size)
        self.packet_jitter_history = np.zeros(self.history_window_size)
        # Loss Metrics
        self.packet_loss_ratio_history = np.zeros(self.history_window_size)
        self.average_lost_packets_history = np.zeros(self.history_window_size)
        # Packet Type metrics
        self.video_packet_probability_history = np.zeros(self.history_window_size)
        self.audio_packet_probability_history = np.zeros(self.history_window_size)
        self.probing_packet_probability_history = np.zeros(self.history_window_size)
        # Action metrics
        self.previous_actions_history = np.zeros(self.history_window_size)
        # # Feedback metrics
        # self.timesteps_since_last_feedback = np.zeros(self.history_window_size)
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path_dir = "/opt/home_dir/AlphaRTC/scripts/models/"
        self.models = self.load_models(model_path_dir)
        # indices of models to use
        self.model_indices = np.arange(len(self.models))

    def load_models(self, model_path):
        # get all .pth files in the directory
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        models = []
        for model_file in model_files:
            model = Actor(120, 1)
            model.load_state_dict(torch.load(model_path + model_file, map_location=torch.device('cpu')))
            model = model.to(self.device)
            model.eval()
            models.append(model)
        return models
        
    def report_states(self, stats: dict):
        '''
        stats is a dict with the following items
        {
            "send_time_ms": uint,
            "arrival_time_ms": uint,
            "payload_type": int,
            "sequence_number": uint,
            "ssrc": int,
            "padding_length": uint,
            "header_length": uint,
            "payload_size": uint
        }
        '''
        packet = PacketInfo()
        packet.payload_size = stats["payload_size"]
        packet.arrival_time_ms = stats["arrival_time_ms"]
        packet.send_time_ms = stats["send_time_ms"]
        packet.transport_seq_num = stats["sequence_number"]
        packet.is_video = self.is_video_payload(stats["payload_type"])
        packet.is_audio = self.is_audio_packet(stats["payload_type"])
        packet.is_probing = self.is_probing_packet(stats["payload_type"])
        self.packet_queue.append(packet)

    def get_estimated_bandwidth(self)->int:
        self.process_features()
        state = self.get_state()
        # choose random model
        model_idx = np.random.choice(self.model_indices)
        model = self.models[model_idx]
        with torch.no_grad():
            self.bwe = model(state)
        with open("/opt/home_dir/AlphaRTC/scripts/estimator_debug.log", "a") as f:
            # f.write(f'{state}\n')
            f.write(f'{self.bwe}\n')
        self.bwe = self.log_to_linear(self.bwe.item())
        return int(self.bwe)   

    def log_to_linear(self, log_action: float)->float:
        min_mbps = self.min_bwe / 1e6
        max_mbps = self.max_bwe / 1e6
        log_action = np.clip(log_action, 0, 1)
        log_bwe_mbps = log_action * (self.log_max_mbps - self.log_min_mbps) + self.log_min_mbps
        bwe_mbps = np.clip(np.exp(log_bwe_mbps), min_mbps, max_mbps)
        bwe_bps = int(bwe_mbps * 1e6)
        return bwe_bps
    
    def get_state(self):
        # NOTE: The order of the arrays is important
        state = np.column_stack([
            self.audio_packet_probability_history,
            self.average_lost_packets_history,
            self.delay_avg_min_difference_history,
            self.delay_ratio_history,
            self.min_seen_delay_history,
            self.delay_history,
            self.packet_interarrival_time_history,
            self.packet_jitter_history,
            self.packet_loss_ratio_history,
            self.queuing_delay_history,
            self.receiving_rate_history,
            self.video_packet_probability_history,
        ])
        # shape of state is (1 x 10 * 12)
        state = state.reshape(1, -1)
        state = torch.from_numpy(state).float()
        return state

    def process_features(self):
        """Processes metrics from the packet queue and updates history arrays"""        
        if len(self.packet_queue) == 0:
            # No packets to process
            return
        
        # Initialize counters and data structures for this interval
        total_bytes = 0
        total_packets = 0
        video_packets = 0
        audio_packets = 0
        probing_packets = 0
        
        # Delay tracking
        sum_delay_ms = 0
        min_delay_ms_this_interval = float('inf')
        
        # For interarrival calculation
        arrival_times = []
        
        # For loss detection
        received_seq_nums = {}
        min_seq = float('inf')
        max_seq = -1
        
        # Process all packets in the queue
        while len(self.packet_queue) > 0:
            packet = self.packet_queue.pop(0)
            
            # Basic packet statistics
            total_bytes += packet.payload_size
            total_packets += 1
            
            # Packet type counting
            if packet.is_video:
                video_packets += 1
            if packet.is_audio:
                audio_packets += 1
            if packet.is_probing:
                probing_packets += 1
            
            # Store arrival time for interarrival calculation
            arrival_times.append(packet.arrival_time_ms)
            
            # Calculate packet delay
            packet_delay_ms = packet.arrival_time_ms - packet.send_time_ms
            packet_delay_ms = min(packet_delay_ms, 1000)  # Cap at 1 second
            if packet_delay_ms >= 0:  # Ignore negative delays
                sum_delay_ms += packet_delay_ms
                
                # Update minimum delays
                if packet_delay_ms < self.min_delay_ms_overall:
                    self.min_delay_ms_overall = packet_delay_ms
                if packet_delay_ms < min_delay_ms_this_interval:
                    min_delay_ms_this_interval = packet_delay_ms
            
            # Track sequence numbers for loss calculation
            if packet.transport_seq_num != -1:
                received_seq_nums[packet.transport_seq_num] = True
                min_seq = min(min_seq, packet.transport_seq_num)
                max_seq = max(max_seq, packet.transport_seq_num)
        
        # Calculate receiving rate
        interval_duration_sec = self.measurement_interval_ms / 1000.0
        receiving_rate_bps = int(total_bytes * 8 / interval_duration_sec)
        
        # Calculate average delay
        avg_delay_ms = (sum_delay_ms / total_packets) if total_packets > 0 else 0
        
        # Calculate queuing delay
        queuing_delay_ms = (avg_delay_ms - self.min_delay_ms_overall) if total_packets > 0 else 0
        
        # Calculate delay ratio
        delay_ratio = (avg_delay_ms / min_delay_ms_this_interval) if (min_delay_ms_this_interval != float('inf') and min_delay_ms_this_interval > 0) else 1.0
        
        # Calculate delay average/min difference
        delay_avg_min_difference_ms = (avg_delay_ms - min_delay_ms_this_interval) if min_delay_ms_this_interval != float('inf') else 0
        
        # Calculate interarrival metrics
        mean_interarrival_ms = 0
        jitter_ms = 0
        
        if len(arrival_times) > 1:
            arrival_times.sort()
            
            interarrival_times = []
            for i in range(1, len(arrival_times)):
                interarrival_times.append(arrival_times[i] - arrival_times[i-1])
            
            # Calculate mean
            mean_interarrival_ms = sum(interarrival_times) / len(interarrival_times)
            
            # Calculate standard deviation (jitter)
            sq_sum = sum((time - mean_interarrival_ms) ** 2 for time in interarrival_times)
            jitter_ms = np.sqrt(sq_sum / len(interarrival_times))
        
        # Calculate loss metrics
        lost_packets = 0
        packet_loss_ratio = 0
        average_lost_packets = 0
        
        if min_seq != float('inf') and max_seq != -1:
            # Calculate expected number of packets
            expected_packets = max_seq - min_seq + 1
            
            # Calculate lost packets
            lost_packets = expected_packets - len(received_seq_nums)
            
            # Calculate loss ratio
            packet_loss_ratio = (lost_packets / expected_packets) if expected_packets > 0 else 0
            
            # For average lost packets, we'd need to track loss bursts
            # This is a simplified version
            average_lost_packets = lost_packets
        
        # Calculate packet type probabilities
        video_packets_probability = (video_packets / total_packets) if total_packets > 0 else 0
        audio_packets_probability = (audio_packets / total_packets) if total_packets > 0 else 0
        probing_packets_probability = (probing_packets / total_packets) if total_packets > 0 else 0
        
        # Store computed metrics
        # Rate metrics (normalized)
        receiving_rate_bps = np.clip(receiving_rate_bps, self.min_bwe, self.max_bwe) / self.max_bwe
        self.update_metric(self.receiving_rate_history, receiving_rate_bps)
        
        # Delay metrics (normalized)
        queuing_delay_ms = np.clip(queuing_delay_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.queuing_delay_history, queuing_delay_ms)
        avg_delay_ms = np.clip(avg_delay_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.delay_history, avg_delay_ms)
        self.min_seen_delay_ms_overall = np.clip(self.min_delay_ms_overall, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.min_seen_delay_history, self.min_delay_ms_overall)
        delay_avg_min_difference_ms = np.clip(delay_avg_min_difference_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.delay_avg_min_difference_history, delay_avg_min_difference_ms)
        delay_ratio = np.clip(delay_ratio - 1.0, 0, 1)  # avg delay should be greater than min delay
        self.update_metric(self.delay_ratio_history, delay_ratio)

        # Jitter metrics
        mean_interarrival_ms = np.clip(mean_interarrival_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.packet_interarrival_time_history, mean_interarrival_ms)
        jitter_ms = np.clip(jitter_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.packet_jitter_history, jitter_ms)
        
        # Packet loss metrics
        packet_loss_ratio = np.clip(packet_loss_ratio, 0, 1)
        self.update_metric(self.packet_loss_ratio_history, packet_loss_ratio)
        average_lost_packets = np.clip(average_lost_packets, 0, self.max_lost_packets) / self.max_lost_packets
        self.update_metric(self.average_lost_packets_history, average_lost_packets)
        
        # Packet type metrics
        self.update_metric(self.video_packet_probability_history, video_packets_probability)
        self.update_metric(self.audio_packet_probability_history, audio_packets_probability)
        self.update_metric(self.probing_packet_probability_history, probing_packets_probability)
        
        # Misc metrics
        # timesteps_since_feedback = int((now_ms - self.last_feedback_report_ms) / self.measurement_interval_ms)
        # self.update_metric(self.timesteps_since_last_feedback, timesteps_since_feedback)
        # print(f"Timesteps since last feedback: {self.vector_to_string(self.timesteps_since_last_feedback)}")
        
        # Previous actions would be updated elsewhere in the code

    def update_metric(self, metric_history, new_value):
        """Updates a metric history array with a new value"""
        metric_history[:-1] = metric_history[1:]  # Shift values to the left
        metric_history[-1] = new_value  # Add new value at the end

    def is_probing_packet(self, payload_type):
        """Determines if a packet is a probing packet"""
        return False        
    
    def is_audio_packet(self, payload_type):
        # from SDP
        """Determines if a packet is a audio packet"""
        if (payload_type == 111 or payload_type == 103 or 
            payload_type == 104 or payload_type == 9 or
            payload_type == 102 or payload_type == 0 or
            payload_type == 8 or payload_type == 106 or
            payload_type == 105 or payload_type == 13 or
            payload_type == 110 or payload_type == 112 or
            payload_type == 113 or payload_type == 126):
            return True
        return False
    
    def is_video_payload(self, payload_type):
        # using VP9
        if (payload_type == 96 or payload_type == 97 or 
            payload_type == 98 or payload_type == 99 or
            payload_type == 100 or payload_type == 101 or
            payload_type == 127 or payload_type == 123 or
            payload_type == 125 or payload_type == 122 or
            payload_type == 124):
            return True
        return False