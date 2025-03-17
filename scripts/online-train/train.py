import numpy as np
import os
import ppo2 as network
import torch
import pickle
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = 70
A_DIM = 5
ACTOR_LR_RATE = 1e-4
RANDOM_SEED = 42
# TODO: set the output directory
OUTPUT_DIR = './ppo'
MODEL_DIR = './models'
LOG_FILE = OUTPUT_DIR + '/log'
CHECKPOINT_DIR = OUTPUT_DIR + '/checkpoints'

TRAJECORY_PATH = "/Users/silver/Desktop/alphartc_process_logs/data/meta_trajectories/meta_trajectories.pkl"

# Create result directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

NN_MODEL = None

def load_trajectory():   
    # Load trajectory data
    with open(TRAJECORY_PATH, 'rb') as f:
        data = pickle.load(f)
    
    states = []
    rewards = []
    actions = []

    key = list(data.keys())[0]

    # NOTE: We should have 1 key in the dictionary
    data = data[key]
    sender_t_steps = len(data['sender_data']['states'])
    receiver_t_steps = len(data['receiver_data']['states'])
    if sender_t_steps >= 15:
        states.extend(data['sender_data']['states'])
        rewards.extend(data['sender_data']['rewards'][:sender_t_steps])
        actions.extend(data['sender_data']['actions'][:sender_t_steps])
    if receiver_t_steps >= 15:
        states.extend(data['receiver_data']['states'])
        rewards.extend(data['receiver_data']['rewards'][:receiver_t_steps])
        actions.extend(data['receiver_data']['actions'][:receiver_t_steps])
    # NOTE: if < 15 then error occurred during call so just skip
    assert sender_t_steps >= 15 or receiver_t_steps >= 15
    return states, rewards, actions

def train_agent(epoch_num):
    # Load trajectory data
    states, rewards, actions = load_trajectory()

    with open(LOG_FILE + '_train.txt', 'w') as train_log_file:
        # Create environment and actor network
        actor = network.Network(
            state_dim=S_DIM,
            action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE
        )

        # Restore neural net parameters if available
        try:
            if NN_MODEL is not None:
                actor.load_model(NN_MODEL)
                print('Model restored.')
        except Exception as e:
            print('Starting from scratch:', e)
            
        # Collect experience
        obs = states[0]
        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        done = False

        # Loop through trajectory
        for i in range(1, len(states)):
            s_batch.append(obs)
            
            # Get action from policy
            action_prob = actor.predict(obs)
            action = actions[i]
            
            # Take action in environment
            obs = states[i]
            
            # Store experience
            action_vec = np.zeros(A_DIM)
            action_vec[action] = 1
            a_batch.append(action_vec)
            r_batch.append(rewards[i])
            p_batch.append(action_prob)
            
        # Calculate values
        done = True
        v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
        
        # Convert to numpy arrays
        s_batch = np.stack(s_batch, axis=0)
        a_batch = np.vstack(a_batch)
        p_batch = np.vstack(p_batch)
        v_batch = np.vstack(v_batch)
        
        # Train the network
        actor.train(s_batch, a_batch, p_batch, v_batch)
        
        # Log training data
        avg_reward = np.mean(rewards)
        train_log_file.write(f'Epoch: {epoch_num}, Average Reward: {avg_reward:.2f}\n')
        print(f'Epoch: {epoch_num}, Average Reward: {avg_reward:.2f}')
        
        # Save the model
        actor.save_model(f'{CHECKPOINT_DIR}/model_n_call_{epoch_num * 2}.pth')
    print("Training completed!")

def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--epoch', type=int, default=1, help='Current epoch training number')
    return parser.parse_args()

def main():
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)

    args = parse_args()
    
    # Set number of epochs for training
    train_agent(args.epoch)

if __name__ == '__main__':
    main()