import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

import pickle
import os
import time



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
    

class MetaActor(nn.Module):
    def __init__(self, action_dim=5, input_dim=70, hidden_size=128, action_eps=1e-4, gamma=0.99, eps=0.2):
        super(MetaActor, self).__init__()
        # Actor network
        self.a_dim = action_dim
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.action_eps = action_eps
        self.gamma = gamma
        self.eps = eps

        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, action_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.out(x), dim=-1)
        a = torch.clamp(x, self.action_eps, 1. - self.action_eps)
        return a


class Critic(nn.Module):
    def __init__(self, action_dim=1, input_dim=70, hidden_size=128, action_eps=1e-4, gamma=0.99, eps=0.2):
        super(Critic, self).__init__()
        # Critic network
        self.a_dim = action_dim
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.action_eps = action_eps
        self.gamma = gamma
        self.eps = eps

        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.out(x)
        return v
    
class Network():
    def __init__(self, state_dim=70, action_dim=5, learning_rate=1e-3, eps=0.2, gamma=0.99):

        self.s_dim = state_dim
        self.action_dim = action_dim
        self._entropy_weight = np.log(action_dim)
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 1
        self.eps = eps
        self.gamma = gamma

        self.actor = MetaActor()
        self.critic = Critic()
        self.lr_rate = learning_rate
        self.optimizer = optim.Adam(list(self.actor.parameters()) + \
                                    list(self.critic.parameters()), lr=learning_rate)

    def get_network_params(self):
        return [self.actor.state_dict(), self.critic.state_dict()]
    
    def set_network_params(self, input_network_params):
        actor_net_params, critic_net_params = input_network_params
        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)

    def r(self, pi_new, pi_old, acts):
        return torch.sum(pi_new * acts, dim=1, keepdim=True) / \
               torch.sum(pi_old * acts, dim=1, keepdim=True)

    def train(self, s_batch, a_batch, p_batch, v_batch):
        s_batch = torch.from_numpy(s_batch).to(torch.float32)
        a_batch = torch.from_numpy(a_batch).to(torch.float32)
        p_batch = torch.from_numpy(p_batch).to(torch.float32)
        v_batch = torch.from_numpy(v_batch).to(torch.float32)

        for _ in range(self.PPO_TRAINING_EPO):
            pi = self.actor.forward(s_batch)
            val = self.critic.forward(s_batch)

            # loss
            adv = v_batch - val.detach()
            ratio = self.r(pi, p_batch, a_batch)
            ppo2loss = torch.min(ratio * adv, torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv)
            # Dual-PPO
            dual_loss = torch.where(adv < 0, torch.max(ppo2loss, 3. * adv), ppo2loss)
            loss_entropy = torch.sum(-pi * torch.log(pi), dim=1, keepdim=True)

            loss = -dual_loss.mean() + 10. * F.mse_loss(val, v_batch) - self._entropy_weight * loss_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update entropy weight
        _H = (-(torch.log(p_batch) * p_batch).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)

    def predict(self, input):
        with torch.no_grad():
            pi = self.actor.forward(input)[0]
            return pi.numpy()

    def load_model(self, nn_model):
        actor_model_params, critic_model_params = torch.load(nn_model, map_location=torch.device('cpu'))
        self.actor.load_state_dict(actor_model_params)
        self.critic.load_state_dict(critic_model_params)
        self.actor.eval()
        self.critic.eval()

    def save_model(self, nn_model):
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        torch.save(model_params, nn_model)

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        R_batch = np.zeros_like(r_batch)

        if terminal:
            # in this case, the terminal reward will be assigned as r_batch[-1]
            R_batch[-1] = r_batch[-1]  # terminal state
        else:
            val = self.critic.forward(s_batch)
            R_batch[-1] = val[-1]  # bootstrap from last state

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t] = r_batch[t] + self.gamma * R_batch[t + 1]

        return list(R_batch)
           
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
        # Metapolicy models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path_dir = "/opt/home_dir/AlphaRTC/scripts/models/"
        self.models = self.load_models(model_path_dir)
        # indices of models to use
        self.model_indices = np.arange(len(self.models))
        self.previous_decision = 0
        # meta constants
        self.meta_counter = 1
        self.meta_feature_update_interval = 10
        self.meta_decision_interval = 100  # 6 seconds
        # meta fetures
        self.meta_model = self.load_meta_model()
        self.meta_packet_queue = []
        self.meta_receiving_rate_history = np.zeros(self.history_window_size)
        self.meta_delay_history = np.zeros(self.history_window_size)
        self.meta_average_lost_packets_history = np.zeros(self.history_window_size)
        self.meta_packet_interarrival_time_history = np.zeros(self.history_window_size)
        self.meta_video_packet_probability_history = np.zeros(self.history_window_size)
        self.meta_audio_packet_probability_history = np.zeros(self.history_window_size)
        self.meta_previous_actions_history = np.zeros(self.history_window_size)
        # for offline training purposes
        self.id = int(time.time_ns()) # NOTE: sender id is always less than receiver id
        self.meta_trajectory = {
            'states': [],
            'actions': []
        }


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
    
    def load_meta_model(self):
        model = Network()
        model.load_model("/opt/home_dir/AlphaRTC/scripts/meta_model/meta.pth")
        return model
        
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
        self.meta_packet_queue.append(packet)

    def get_estimated_bandwidth(self)->int:
        self.process_features(self.packet_queue)
        self.packet_queue = []
        state = self.get_state()
        model = self.handle_model_selection()
        with torch.no_grad():
            self.bwe = model(state)
        self.bwe = self.log_to_linear(self.bwe.item())
        self.meta_counter += 1
        return int(self.bwe)   

    def handle_model_selection(self):
        if self.meta_counter % self.meta_feature_update_interval == 0:
            self.process_features(self.meta_packet_queue, is_meta=True)
            self.meta_packet_queue = []
        # every 6 seconds, make a decision
        if self.meta_counter % self.meta_decision_interval == 0:
            self.meta_model = self.load_meta_model()  # load latest weights
            meta_state = self.get_meta_state()
            # choose random model (for now)
            # model_idx = np.random.choice(self.model_indices)
            model_idx = np.argmax(self.meta_model.predict(meta_state))
            # NOTE: For offline training purposes
            self.meta_trajectory['states'].append(meta_state)
            self.meta_trajectory['actions'].append(model_idx)
            with open(f"/mydata/meta_trajectories/{self.id}.pkl", "wb") as f:
                pickle.dump(self.meta_trajectory, f)
            # update previous decision
            self.meta_counter = 1
            self.previous_decision = model_idx
            self.meta_previous_actions_history[:-1] = self.meta_previous_actions_history[1:]
            self.meta_previous_actions_history[-1] = model_idx / len(self.models)  # make it [0, 1]
        else:
            model_idx = self.previous_decision
        model = self.models[model_idx]
        return model

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
    
    def get_meta_state(self):
        state = np.column_stack([
            self.meta_audio_packet_probability_history,
            self.meta_average_lost_packets_history,
            self.meta_delay_history,
            self.meta_packet_interarrival_time_history,
            self.meta_receiving_rate_history,
            self.meta_video_packet_probability_history,
            self.meta_previous_actions_history
        ])
        state = state.reshape(1, -1)
        state = torch.from_numpy(state).float()
        return state

    def process_features(self, packet_queue, is_meta=False):
        """Processes metrics from the packet queue and updates history arrays"""        
        if len(packet_queue) == 0:
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
        while len(packet_queue) > 0:
            packet = packet_queue.pop(0)
            
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
        if is_meta:
            interval_duration_sec = (self.meta_feature_update_interval * self.measurement_interval_ms) / 1000.0
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
        if is_meta:
            self.update_meta_metrics(receiving_rate_bps, avg_delay_ms, mean_interarrival_ms,
                                     average_lost_packets, video_packets_probability,
                                     audio_packets_probability)
        else:
            self.update_metrics(receiving_rate_bps, queuing_delay_ms, avg_delay_ms, min_delay_ms_this_interval,
                                delay_avg_min_difference_ms, delay_ratio, mean_interarrival_ms, jitter_ms,
                                packet_loss_ratio, average_lost_packets, video_packets_probability,
                                audio_packets_probability, probing_packets_probability)

    def update_meta_metrics(self, receiving_rate_bps, avg_delay_ms, mean_interarrival_ms,
                            average_lost_packets, video_packets_probability,
                            audio_packets_probability):        
        """Updates all meta metrics with the new values"""
        # Rate metrics (normalized)
        receiving_rate_bps = np.clip(receiving_rate_bps, self.min_bwe, self.max_bwe) / self.max_bwe
        self.update_metric(self.meta_receiving_rate_history, receiving_rate_bps)
        # Delay metrics (normalized)    
        avg_delay_ms = np.clip(avg_delay_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.meta_delay_history, avg_delay_ms)
        # Jitter metrics (normalized)
        mean_interarrival_ms = np.clip(mean_interarrival_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.meta_packet_interarrival_time_history, mean_interarrival_ms)
        # Loss metrics (normalized)
        average_lost_packets = np.clip(average_lost_packets, 0, self.max_lost_packets) / self.max_lost_packets
        self.update_metric(self.meta_average_lost_packets_history, average_lost_packets)
        # Media type metrics
        self.update_metric(self.meta_video_packet_probability_history, video_packets_probability)
        self.update_metric(self.meta_audio_packet_probability_history, audio_packets_probability)


    def update_metrics(self, receiving_rate_bps, queuing_delay_ms, avg_delay_ms, min_delay_ms_this_interval,
                          delay_avg_min_difference_ms, delay_ratio, mean_interarrival_ms, jitter_ms,
                            packet_loss_ratio, average_lost_packets, video_packets_probability,
                            audio_packets_probability, probing_packets_probability):
        """Updates all metrics with the new values"""
        # Rate metrics (normalized)
        receiving_rate_bps = np.clip(receiving_rate_bps, self.min_bwe, self.max_bwe) / self.max_bwe
        self.update_metric(self.receiving_rate_history, receiving_rate_bps)
        # Delay metrics (normalized)
        queuing_delay_ms = np.clip(queuing_delay_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.queuing_delay_history, queuing_delay_ms)
        avg_delay_ms = np.clip(avg_delay_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.delay_history, avg_delay_ms)
        self.min_seen_delay_ms_overall = np.clip(self.min_delay_ms_overall, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.min_seen_delay_history, min_delay_ms_this_interval)
        delay_avg_min_difference_ms = np.clip(delay_avg_min_difference_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.delay_avg_min_difference_history, delay_avg_min_difference_ms)
        delay_ratio = np.clip(delay_ratio - 1.0, 0, 1)  # avg delay should be greater than min delay
        self.update_metric(self.delay_ratio_history, delay_ratio)
        # Jitter metrics (normalized)
        mean_interarrival_ms = np.clip(mean_interarrival_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.packet_interarrival_time_history, mean_interarrival_ms)
        jitter_ms = np.clip(jitter_ms, 0, self.max_delay_ms) / self.max_delay_ms
        self.update_metric(self.packet_jitter_history, jitter_ms)
        # Loss metrics (normalized)
        packet_loss_ratio = np.clip(packet_loss_ratio, 0, 1)
        self.update_metric(self.packet_loss_ratio_history, packet_loss_ratio)
        average_lost_packets = np.clip(average_lost_packets, 0, self.max_lost_packets) / self.max_lost_packets
        self.update_metric(self.average_lost_packets_history, average_lost_packets)
        # Media type metrics
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