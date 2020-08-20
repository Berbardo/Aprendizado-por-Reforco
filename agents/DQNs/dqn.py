import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.experience_replay import NumpyReplay

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DQN:
    def __init__(self, observation_space, action_space, lr=7e-4, gamma=0.99, tau=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.memory = NumpyReplay(1000000, observation_space.shape[0], self.device)
        self.action_space = action_space

        self.epsilon = 0.5
        self.epsilon_decay = 0.9995
        self.min_epsilon = 0.01

        self.tau = tau
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

    def remember(self, state, action, reward, new_state, done):
        for i in range(len(state)):
            self.memory.update(state[i], action[i], reward[i], new_state[i], done[i])

    def train(self, batch_size=128, epochs=1):
        if 100 > self.memory.size:
            return
        
        for epoch in range(epochs):
            (states, actions, rewards, next_states, dones) = self.memory.sample(batch_size)

            actions = actions.unsqueeze(-1)
            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)

            q = self.dqn.forward(states).gather(-1, actions.long())

            with torch.no_grad():
                q2 = self.dqn_target.forward(next_states).max(dim=-1, keepdim=True)[0]

                target = (rewards + (1 - dones) * self.gamma * q2).to(self.device)
            

            loss = F.mse_loss(q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_target()

    def update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
