import numpy as np
from collections import deque
import torch
import argparse
import glob
from utils import save, load_data
import random
from agent import IQL
from torch.utils.data import DataLoader, TensorDataset

from time import time

from torch.utils.tensorboard import SummaryWriter
# set log directory
writer = SummaryWriter("./logs")


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="Local_Meta_IQL", help="Name of the run, default: IQL")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=1, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size, default: 128")
    parser.add_argument("--hidden_size", type=int, default=128, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=3, help="Set to 0 for imitation learning.")
    parser.add_argument("--expectile", type=float, default=0.7, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1000, help="")
    parser.add_argument("--categorical", action='store_true', help="Represent the value function as a categorical distribution")
    args = parser.parse_args()
    return args

def prep_dataloader(batch_size=256, seed=1):
    dataset, min_reward, max_reward = load_data(save_data=True)
    tensors = {}
    for k, v in dataset.items():
        if k in ["actions", "states", "next_states", "rewards", "dones"]:
            if k != "dones":  
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()
    tensordata = TensorDataset(tensors["states"],
                               tensors["actions"],
                               tensors["rewards"],
                               tensors["next_states"],
                               tensors["dones"])
    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    return dataloader, min_reward, max_reward

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # NOTE: we need the min and max reward for the HLGaussLoss
    dataloader, min_reward, max_reward = prep_dataloader(batch_size=config.batch_size, seed=config.seed)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # FOR Apple Silicon
    device = torch.device("mps")

    # TODO: hardcoded for now
    state_dim = 70
    action_dim = 5

    agent = IQL(state_size=state_dim,
                action_size=action_dim,
                learning_rate=config.learning_rate,
                hidden_size=config.hidden_size,
                tau=config.tau,
                temperature=config.temperature,
                expectile=config.expectile,
                device=device)
        
    start_time = time()
    actor_losses = []
    for i in range(1, config.episodes+1):
        metrics = {
            "avg_policy_loss": 0,
            "avg_value_loss": 0,
            "avg_critic1_loss": 0,
            "avg_critic2_loss": 0,
            "avg_value": 0,
            "avg_q1": 0,
            "avg_q2": 0,
            # debug metrics
            'avg_policy_entropy': 0
        }
        batches = 0
        for batch_idx, experience in enumerate(dataloader):
            states, actions, rewards, next_states, dones = experience

            # flatten states for MLP
            states = states.view(-1, state_dim).to(device)
            # get indices of non-padded elements
            mask = (states != -999).all(dim=-1)  # NOTE: -999 is the padding value. We lose this info after processing rewards. So we need to maintain these indices through the states
            actions = actions.view(-1, action_dim).to(device)
            rewards = rewards.view(-1, 1).to(device)
            next_states = next_states.view(-1, state_dim).to(device)
            dones = dones.view(-1, 1).to(device)
            policy_loss, critic1_loss, critic2_loss, value_loss, value, q1, q2, misc = agent.learn((states, actions, rewards, next_states, dones), mask)
            batches += 1

            # update metrics
            metrics["avg_policy_loss"] += policy_loss
            metrics["avg_value_loss"] += value_loss
            metrics["avg_critic1_loss"] += critic1_loss
            metrics["avg_critic2_loss"] += critic2_loss
            metrics["avg_value"] += value
            metrics["avg_q1"] += q1
            metrics["avg_q2"] += q2

            # debug metrics
            metrics['avg_policy_entropy'] += misc['policy_entropy']

        # log metrics
        actor_losses.append(metrics["avg_policy_loss"] / batches)
        for k, v in metrics.items():
            writer.add_scalar(k, v / batches, i)

        if i % 50 == 0:
            print(f'Completed {i} Episodes. Time Elapsed: {(time() - start_time) / 60.0:.2f} Minutes on {device}')

        if i % config.save_every == 0:
            save(config, model=agent.actor_local, ep=i)
            # save(config, model=agent.value_net, ep=i, value=True)

    print(f'Completed {config.episodes} Episodes. Total Time Elapsed: {(time() - start_time) / 60.0:.2f} Minutes on {device}')
    actor_losses = np.array(actor_losses)
    print(f'Training Complete. Best Actor Checkpoint: {actor_losses.argmin()} with Loss: {actor_losses.min()}')


if __name__ == "__main__":
    config = get_config()
    train(config)