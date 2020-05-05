import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from noisy_linear import NoisyLinear
from experience_replay import ReplayBuffer
from prioritized_replay import PrioritizedReplayBuffer

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, atom_dim: int, support):
        super(Network, self).__init__()
        self.out_dim = out_dim
        self.atom_dim = atom_dim
        self.support = support

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        self.advantage1 = NoisyLinear(128, 128)
        self.advantage2 = NoisyLinear(128, out_dim * atom_dim)

        self.value1 = NoisyLinear(128, 128)
        self.value2 = NoisyLinear(128, atom_dim)

    def forward(self, state):
        dist = self.dist(state)
        q = torch.sum(dist * self.support, dim=2)

        return q    

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        
        value = F.relu(self.value1(feature))
        value = self.value2(value).view(-1, 1, self.atom_dim)
        
        advantage = F.relu(self.advantage1(feature))
        advantage = self.advantage2(advantage).view(-1, self.out_dim, self.atom_dim)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage1.reset_noise()
        self.advantage2.reset_noise()
        self.value1.reset_noise()
        self.value2.reset_noise()

class Rainbow:
    def __init__(self, observation_space, action_space, lr=7e-4, gamma=0.99, tau=0.01, n_step=3, n_envs=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.beta = 0.6
        self.n_step = n_step
        self.memory = PrioritizedReplayBuffer(10000, 0.5, n_step, n_envs, gamma)
        self.action_space = action_space

        self.v_min = -10
        self.v_max = 10
        self.atom_size = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        self.update_count = 0
        self.dqn = Network(observation_space.shape[0], action_space.n, self.atom_size, self.support).to(self.device)
        self.dqn_target = Network(observation_space.shape[0], action_space.n, self.atom_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer  = optim.Adam(self.dqn.parameters(), lr=lr)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.dqn.forward(state).argmax(dim=-1)
        action = action.cpu().detach().numpy()

        return action

    def remember(self, states, actions, rewards, new_states, dones):
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], new_states[i], dones[i], i)

    def train(self, batch_size=32):
        if 500 > len(self.memory._storage):
            return
        
        self.update_count +=1

        self.beta = self.beta + self.update_count/100000 * (1.0 - self.beta)

        (states, actions, rewards, next_states, dones, weights, batch_indexes) = self.memory.sample(batch_size, self.beta)
        weights = torch.FloatTensor(weights).unsqueeze(-1).to(self.device)

        td_error = self.calculate_loss(states, actions, rewards, next_states, dones, self.gamma** self.n_step)

        # gamma = self.gamma ** self.n_step
        # (states, actions, rewards, next_states, dones) = self.memory_n.sample_batch_from_idxs(batch_indexes)
        # n_loss = self.calculate_loss(states, actions, rewards, next_states, dones, gamma)

        # td_error += n_loss
        loss = torch.mean(td_error * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        self.update_target()

        priorities = td_error.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(batch_indexes, priorities)

        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

    def calculate_loss(self, states, actions, rewards, next_states, dones, gamma):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn_target.forward(next_states).argmax(dim=1)
            next_dist = self.dqn_target.dist(next_states)
            next_dist = next_dist[range(len(states)), next_action]

            t_z = rewards + (1 - dones) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (len(states) - 1) * self.atom_size, len(states))
            offset = offset.long().unsqueeze(1).expand(len(states), self.atom_size).to(self.device)

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.dqn.dist(states)
        log_p = torch.log(dist[range(len(states)), actions])
        td_error = -(proj_dist * log_p).sum(1)

        return td_error

    def update_target(self):
        for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def hard_update_target(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
