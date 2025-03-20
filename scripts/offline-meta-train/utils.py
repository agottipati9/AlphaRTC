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

def preprocess_action(actions):
    action_vectors = []
    for action in actions:
        action_vector = np.zeros(5)
        action_vector[action] = 1
        action_vectors.append(action_vector)
    actions = np.array(action_vectors)
    return actions

def preprocess_data(data):
    all_call_states = []
    all_call_next_states = []
    all_call_actions = []
    all_call_rewards = []
    all_call_dones = []
    # get all states and actions
    for i, call in enumerate(data):
        call_data = data[call]
        # append sender data
        sender_data = call_data['sender_data']
        all_call_states.append(torch.stack(sender_data['states']).numpy()[:-1])
        all_call_next_states.append(torch.stack(sender_data['states']).numpy()[1:])
        all_call_actions.append(preprocess_action(sender_data['actions'])[1:])
        # trim rewards
        rewards = sender_data['rewards']
        rewards = rewards[:len(sender_data['states'])]  # trim to match states
        all_call_rewards.append(np.array(rewards).reshape(-1, 1)[1:])
        all_call_dones.append(np.array([0] * len(sender_data['states'])).reshape(-1, 1)[1:])
        all_call_dones[i][-1] = 1  # last done is 1 for sender data
        assert len(all_call_states[-1]) == len(all_call_actions[-1]) == len(all_call_rewards[-1]) == len(all_call_dones[-1])
        # append receiver data
        receiver_data = call_data['receiver_data']
        all_call_states.append(torch.stack(receiver_data['states']).numpy()[:-1])
        all_call_next_states.append(torch.stack(receiver_data['states']).numpy()[1:])
        all_call_actions.append(preprocess_action(receiver_data['actions'])[1:])
        # trim rewards
        rewards = receiver_data['rewards']
        rewards = rewards[:len(receiver_data['states'])]
        all_call_rewards.append(np.array(rewards).reshape(-1, 1)[1:])
        all_call_dones.append(np.array([0] * len(receiver_data['states'])).reshape(-1, 1)[1:])
        all_call_dones[i][-1] = 1
        assert len(all_call_states[-1]) == len(all_call_actions[-1]) == len(all_call_rewards[-1]) == len(all_call_dones[-1])
    return all_call_states, all_call_next_states, all_call_actions, all_call_rewards, all_call_dones
    

def load_data(save_data=False):
    with open('./data/meta_trajectories.pkl', 'rb') as f:
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
    # MOS is between 1.0 and 5.0
    replay_buffer['rewards'] = np.clip(replay_buffer['rewards'], 1.0, 5.0) / 5.0
    return replay_buffer, 0, 0

def process_rewards(rewards, min_reward, max_reward, mean_reward, std_reward):
    # rewards = clip_rewards(rewards, min_reward, max_reward)
    # MOS is between 1.0 and 5.0
    rewards = np.clip(rewards, 1.0, 5.0) / 5.0
    # rewards = min_max_scaling(rewards, min_reward, max_reward)  # NOTE: used to illustrate deviation in value estimates
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



