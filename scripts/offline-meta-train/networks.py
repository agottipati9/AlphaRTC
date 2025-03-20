import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=70, action_size=5, hidden_size=128, init_w=3e-3, log_std_min=-10, log_std_max=2):
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
        x = F.softmax(self.fc3(x), dim=-1)
        return x
    
    def evaluate(self, state, epsilon=1e-6):
        mu = self.forward(state)
        dist = torch.distributions.Categorical(mu)
        action = dist.sample()
        return action, dist
        
    def get_action(self, state):
        mu = self.forward(state)
        dist = torch.distributions.Categorical(mu, 1)
        action = dist.sample()
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu = self.forward(state)
        mu = torch.argmax(mu, dim=-1)
        return mu.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size=70, action_size=5, hidden_size=128, seed=1):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size=70, hidden_size=128):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)