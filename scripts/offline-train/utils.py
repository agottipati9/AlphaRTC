import torch
import numpy as np
import os
import pickle

from tqdm import tqdm

from sklearn.preprocessing import RobustScaler


def save(args, model, ep=None, value=False):
    save_dir = './checkpoints/' 
    name = args.run_name if not value else args.run_name + '_value'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        path = save_dir + name + str(ep) + ".pth"
    else:
        path = save_dir + name + ".pth"
    torch.save(model.state_dict(), path)
    return path

def preprocess_state_feature(feat, feat_vec):
    new_vec = np.array(feat_vec.copy())
    delay_feats = ['Queuing delay (ms)', 'One Way Delay (ms)', 'Minimum seen OWD delay (ms)', 'Delay average min difference (ms)', 'Packet interarrival time (ms)', 'Packet jitter (ms)']
    ratio_feats = ['Packet loss ratio']
    if feat == 'Receiving rate (bps)':
        # cap feature at 8 Mbps
        new_vec = np.minimum(new_vec, 8e6)
        # normalize features by 8e6
        new_vec = new_vec / 8e6
    elif feat in delay_feats:
        # cap at 1000 ms
        new_vec = np.minimum(new_vec, 1000)
        # floor at 0
        new_vec = np.maximum(new_vec, 0)
        # normalize by 1000 ms
        new_vec = new_vec / 1000
    elif feat == 'Timesteps since last feedback':
        # cap at 1000
        new_vec = np.minimum(new_vec, 1000)
        # normalize by 1000
        new_vec = new_vec / 1000
    elif feat == 'Average lost packets':
        # cap at 100
        new_vec = np.minimum(new_vec, 100)
        # normalize by 100
        new_vec = new_vec / 100
    elif feat in ratio_feats:
        # cap at 1
        new_vec = np.minimum(new_vec, 1)
    elif feat == 'Delay ratio':
        # floor at 0
        new_vec = new_vec - 1  # Delay ratio should always be greater than 1
        new_vec = np.clip(new_vec, 0, 1)
    return new_vec

def preprocess_action_feature(action_bps):
    # log transform to force to 0-1
    min_mbps = 0.1
    max_mbps = 8
    # cap between 0.1 and 8 Mpbs
    action_mbps = action_bps / 1e6
    action_mbps = np.minimum(action_mbps, max_mbps)
    action_mbps = np.maximum(action_mbps, min_mbps)
    action_log_mbps = np.log(action_mbps)
    action = (action_log_mbps - np.log(min_mbps)) / (np.log(max_mbps) - np.log(min_mbps))
    return action

def preprocess_rewards(state_dict):
    # bitrate - loss - delay
    bitrates = np.array([state_dict['Receiving rate (bps)'][i][-1] for i in range(len(state_dict['Receiving rate (bps)']))])
    losses = np.array([state_dict['Packet loss ratio'][i][-1] for i in range(len(state_dict['Packet loss ratio']))])
    delays = np.array([state_dict['One Way Delay (ms)'][i][-1] for i in range(len(state_dict['One Way Delay (ms)']))])
    return 2 * bitrates - 1 * losses - 1 * delays
    

def preprocess_data(data):
    all_call_states = []
    all_call_next_states = []
    all_call_actions = []
    all_call_rewards = []
    all_call_dones = []
    # all state features
    features = list(data[0].keys())
    # features to remove
    features.remove('Previous actions')
    features.remove('Timesteps since last feedback')
    features = sorted(features)
    # get all states and actions
    for i, call in enumerate(data):
        call_states = []
        call_actions = []
        call_rewards = []
        call_dones = []
        # process states
        # Shape should be N, T, D, where N is the number of calls, T is the number of timesteps, and D is the number of features 
        states = []
        lengths = []
        reward_dict = {}
        for feat in features:
            feat_vec = preprocess_state_feature(feat, call[feat])  # returns a numpy array  
            feat_vec = feat_vec.reshape(-1, 10, 1)
            states.append(feat_vec)
            lengths.append(feat_vec.shape[0])
            reward_dict[feat] = feat_vec
        # trim to minimum length
        min_length = min(lengths)
        for i in range(len(states)):
            states[i] = states[i][:min_length]
            reward_dict[features[i]] = reward_dict[features[i]][:min_length]  # works because features are ordered
        call_states = np.concatenate(states, axis=-1)
        call_states = call_states.reshape(-1, 10, len(features))
        # process actions
        call_actions = np.array([preprocess_action_feature(float(call['Previous actions'][i][-1])) for i in range(len(call['Previous actions']))])
        call_actions = call_actions.reshape(-1, 1)
        # process rewards
        call_rewards = preprocess_rewards(reward_dict)
        call_dones = np.array([False] * (min_length))
        call_dones[-1] = True
        call_dones = call_dones.reshape(-1, 1)
        # add entire call to replay buffer
        all_call_states.append(call_states[:-1])
        all_call_next_states.append(call_states[1:])
        all_call_actions.append(call_actions[1:])
        all_call_rewards.append(call_rewards[1:])
        all_call_dones.append(call_dones[1:])
    return all_call_states, all_call_next_states, all_call_actions, all_call_rewards, all_call_dones
    

def load_data(save_data=False):
    with open('./data/data.pkl', 'rb') as f:
        data = pickle.load(f)
    # preprocess all data
    all_call_states, all_call_next_states, all_call_actions, all_call_rewards, all_call_dones = preprocess_data(data)
    # initialize replay buffer
    replay_buffer = {
        'states': [],
        'next_states': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    # get max sequence length and reward statistics
    max_seq_len = 0
    for call in all_call_states:
        max_seq_len = max(max_seq_len, call.shape[0])
    # all_rewards_flat = np.concatenate(all_call_rewards)
    # min_reward = np.min(all_rewards_flat)
    # max_reward = np.max(all_rewards_flat)
    # mean_reward = np.mean(all_rewards_flat)
    # std_reward = np.std(all_rewards_flat)
    print('Max sequence length:', max_seq_len)
    # pad everything
    pad_value = -999
    for i in tqdm(range(len(all_call_states))):
        call_states = all_call_states[i]
        call_next_states = all_call_next_states[i]
        call_actions = all_call_actions[i]
        call_rewards = all_call_rewards[i]
        call_dones = all_call_dones[i]
        seq_len = call_states.shape[0]
        pad_len = max_seq_len - seq_len
        replay_buffer['states'].append(np.pad(call_states, ((0, pad_len), (0, 0), (0, 0)), mode='constant', constant_values=pad_value))
        replay_buffer['next_states'].append(np.pad(call_next_states, ((0, pad_len), (0, 0), (0, 0)), mode='constant', constant_values=pad_value))
        replay_buffer['actions'].append(np.pad(call_actions, ((0, pad_len), (0, 0)), mode='constant', constant_values=pad_value))
        replay_buffer['rewards'].append(np.pad(call_rewards, ((0, pad_len), (0, 0)), mode='constant', constant_values=pad_value))
        replay_buffer['dones'].append(np.pad(call_dones, ((0, pad_len), (0, 0)), mode='constant', constant_values=pad_value))
    # convert to numpy arrays
    for k, v in replay_buffer.items():
        replay_buffer[k] = np.array(v)
    
    # save replay buffer as npy
    if save_data:
        np.save('./data/replay_buffer.npy', replay_buffer)
        print('Saved replay buffer to ./data/replay_buffer.npy')

    # TODO: process rewards?
    return replay_buffer, 0, 0

def process_rewards(rewards, min_reward, max_reward, mean_reward, std_reward):
    # rewards = clip_rewards(rewards, min_reward, max_reward)
    rewards = min_max_scaling(rewards, min_reward, max_reward)  # NOTE: used to illustrate deviation in value estimates
    # rewards = normalize_rewards(rewards, mean_reward, std_reward)
    # rewards = log_transform(rewards)  # NOTE: best for distribution learning
    # rewards = log_transform_tanh(rewards)
    # rewards = robust_scale_log_clip_shift(rewards)  # NOTE: best for vanilla IQL
    return rewards

def normalize_rewards(rewards, mean_reward, std_reward):
    return (rewards - mean_reward) / std_reward


def clip_rewards(rewards, min_reward=-1, max_reward=1):
    return np.clip(rewards, min_reward, max_reward)

def log_transform(rewards):
    return np.sign(rewards) * np.log1p(np.abs(rewards))

def log_transform_tanh(rewards):
    return np.tanh(np.sign(rewards) * np.log1p(np.abs(rewards)))

def min_max_scaling(rewards, min_reward, max_reward):
    r = (rewards - min_reward) / (max_reward - min_reward)
    r = 2 * r - 1
    return r

def robust_scale_log_clip_shift(rewards):
    rewards = np.array(rewards)
    scaler = RobustScaler()
    rewards = rewards.flatten().reshape(-1, 1)
    scaler.fit(rewards)
    rewards = scaler.transform(rewards)

    # apply log scaling to rewards: sign(x) * log(1 + abs(x))
    rewards = np.sign(rewards) * np.log1p(np.abs(rewards))

    # clip to [-1, 1]
    rewards = np.clip(rewards, -1, 1)

    # shift > 0.2 to --> 1
    # get indices of rewards > 0.2
    indices = np.where(rewards > -0.2)
    rewards[indices] = rewards[indices] + 1
    return rewards



