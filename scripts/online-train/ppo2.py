import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, action_dim=5, input_dim=70, hidden_size=128, action_eps=1e-4, gamma=0.99, eps=0.2):
        super(Actor, self).__init__()
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
    def __init__(self, state_dim=1, action_dim=5, learning_rate=1e-3, eps=0.2, gamma=0.99):

        self.s_dim = state_dim
        self.action_dim = action_dim
        self._entropy_weight = np.log(action_dim)
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 1
        self.eps = eps
        self.gamma = gamma

        self.actor = Actor()
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

            val = val.squeeze(-1)
            loss = -dual_loss.mean() + 10. * F.mse_loss(val, v_batch) - self._entropy_weight * loss_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update entropy weight
        _H = (-(torch.log(p_batch) * p_batch).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)
        return loss.item()

    def predict(self, input):
        with torch.no_grad():
            pi = self.actor.forward(input)[0]
            return pi.numpy()

    def load_model(self, nn_model):
        actor_model_params, critic_model_params = torch.load(nn_model)
        self.actor.load_state_dict(actor_model_params)
        self.critic.load_state_dict(critic_model_params)

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
           
