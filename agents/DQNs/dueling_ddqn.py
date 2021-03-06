import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.prioritized_replay import PrioritizedReplayBuffer

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Network, self).__init__()

        self.feauture_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, state):
        x = self.feauture_layer(state)
        values = self.value(x)
        advantages = self.advantage(x)

        qvals = values + (advantages - advantages.mean())

        return qvals

class DuelingDDQN:
    def __init__(self, observation_space, action_space, lr=1e-3, gamma=0.99, tau=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.memory = PrioritizedReplayBuffer(100000, 0.6)
        self.action_space = action_space

        self.beta = 0.6
        self.epsilon = 0.7
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.update_count = 0
        self.dqn = Network(observation_space.shape[0], action_space.n).to(self.device)
        self.dqn_target = Network(observation_space.shape[0], action_space.n).to(self.device)
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
            actions = torch.FloatTensor(actions).unsqueeze(-1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(-1).to(self.device)


            q = self.dqn.forward(states).gather(-1, actions.long())
            a2 = self.dqn.forward(next_states).argmax(dim=-1, keepdim=True)
            q2 = self.dqn_target.forward(next_states).gather(-1, a2).detach()

            target = (rewards + (1 - dones) * self.gamma * q2).to(self.device)

            td_error = F.mse_loss(q, target, reduction="none")
            loss = torch.mean(td_error * weights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_target()

            priorities = td_error.detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(batch_indexes, priorities)


    def update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
