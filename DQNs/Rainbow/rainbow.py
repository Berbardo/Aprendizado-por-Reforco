import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from noisy_linear import NoisyLinear
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
        adv_hid = F.relu(self.advantage1(feature))
        val_hid = F.relu(self.value1(feature))
        
        advantage = self.advantage2(adv_hid).view(-1, self.out_dim, self.atom_dim)
        value = self.value2(val_hid).view(-1, 1, self.atom_dim)

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

class Categorical:
    def __init__(self, observation_space, action_space, lr=1e-3, gamma=0.99, tau=0.01, n_step=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.memory = PrioritizedReplayBuffer(100000, 0.5)
        self.action_space = action_space

        self.beta = 0.6
        self.epsilon = 0.7
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.v_min = 0.
        self.v_max = 500.
        self.atom_size = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        self.update_count = 0
        self.dqn = Network(observation_space.shape[0], action_space.n, self.atom_size, self.support).to(self.device)
        self.dqn_target = Network(observation_space.shape[0], action_space.n, self.atom_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer  = optim.Adam(self.dqn.parameters(), lr=lr)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)

        if np.random.random() < self.epsilon:
            action = [self.action_space.sample() for i in range(len(state))]
            return action

        state = torch.FloatTensor(state).to(self.device)
        action = self.dqn.forward(state).argmax(dim=-1)
        action = action.cpu().detach().numpy()

        return action

    def remember(self, states, actions, rewards, new_states, dones):
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], new_states[i], dones[i])

    def train(self, batch_size=32, epochs=1):
        if 1000 > len(self.memory._storage):
            return
        
        for epoch in range(epochs):
            self.update_count +=1

            self.beta = self.beta + self.update_count/100000 * (1.0 - self.beta)

            (states, actions, rewards, next_states, dones, weights, batch_indexes) = self.memory.sample(batch_size, self.beta)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(-1).to(self.device)

            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

            with torch.no_grad():
                next_action = self.dqn_target.forward(next_states).argmax(dim=1)
                next_dist = self.dqn_target.dist(next_states)
                next_dist = next_dist[range(batch_size), next_action]

                t_z = rewards + (1 - dones) * self.gamma * self.support
                t_z = t_z.clamp(min=self.v_min, max=self.v_max)
                b = (t_z - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = torch.linspace(0, (batch_size - 1) * self.atom_size, batch_size)
                offset = offset.long().unsqueeze(1).expand(batch_size, self.atom_size).to(self.device)

                proj_dist = torch.zeros(next_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
                proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )

            dist = self.dqn.dist(states)
            log_p = torch.log(dist[range(batch_size), actions])
            td_error = -(proj_dist * log_p).sum(1)

            loss = torch.mean(td_error * weights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_target()

            priorities = td_error.detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(batch_indexes, priorities)

    def update_target(self):
        for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
