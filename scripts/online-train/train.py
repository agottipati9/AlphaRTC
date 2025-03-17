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
OUTPUT_LOG_DIR = '/mydata/online_train/ppo'
LOG_FILE = OUTPUT_LOG_DIR + '/log'
CHECKPOINT_DIR = OUTPUT_LOG_DIR + '/checkpoints'

# Create result directory
if not os.path.exists(OUTPUT_LOG_DIR):
    os.makedirs(OUTPUT_LOG_DIR)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

def load_trajectory(trajectory_path):   
    # Load trajectory data
    with open(trajectory_path, 'rb') as f:
        data = pickle.load(f)
    
    states = []
    rewards = []
    actions = []

    key = list(data.keys())[0]

    # NOTE: We should have 1 key in the dictionary
    data = data[key]
    sender_t_steps = len(data['sender_data']['states'])
    receiver_t_steps = len(data['receiver_data']['states'])
    # discard tail actions
    states.extend(data['sender_data']['states'])
    rewards.extend(data['sender_data']['rewards'][:sender_t_steps])
    actions.extend(data['sender_data']['actions'][:sender_t_steps])
    # discard tail actions
    states.extend(data['receiver_data']['states'])
    rewards.extend(data['receiver_data']['rewards'][:receiver_t_steps])
    actions.extend(data['receiver_data']['actions'][:receiver_t_steps])
    return states, rewards, actions

def train_agent(args):
    epoch_num = args.epoch
    # Load trajectory data
    states, rewards, actions = load_trajectory(args.traj_path)

    with open(LOG_FILE + '_train.txt', 'a') as train_log_file:
        # Create environment and actor network
        actor = network.Network(
            state_dim=S_DIM,
            action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE
        )

        # Restore neural net parameters if available
        try:
            actor.load_model(args.output_dir)
            print('Model restored.')
        except Exception as e:
            print('Starting from scratch.')
            
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
        loss = actor.train(s_batch, a_batch, p_batch, v_batch)
        
        # Log training data
        avg_reward = np.mean(rewards)
        train_log_file.write(f'Epoch: {epoch_num}, Average Reward: {avg_reward:.2f}, Average Loss: {loss:.2f}\n')
        print(f'Epoch: {epoch_num}, Average Reward: {avg_reward:.2f}, Average Loss: {loss:.2f}')
        
        # Save the model checkpoint
        actor.save_model(f'{CHECKPOINT_DIR}/model_n_call_{epoch_num * 2}.pth')
        # Save the model for inference
        actor.save_model(args.output_dir)
    print(f"Training completed! Model saved to {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--epoch', type=int, default=1, help='Current epoch training number')
    parser.add_argument('--traj_path', type=str, default='/mydata/online_train/meta_trajectories.pkl', help='Path to trajectory data')
    parser.add_argument('--output_dir', type=str, default='/opt/home_dir/AlphaRTC/scripts/meta_model/meta.pth', help='Where to write meta model for inference')
    return parser.parse_args()

def main():
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)

    args = parse_args()

    # check if trajectory path exists
    if not os.path.exists(args.traj_path):
        print('Trajectory path does not exist!')
        return
    
    # Set number of epochs for training
    train_agent(args)

if __name__ == '__main__':
    main()