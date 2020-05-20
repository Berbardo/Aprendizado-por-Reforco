import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.experience_replay import ExperienceReplay

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
    def __init__(self, observation_space, action_space, lr=7e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.memory = ExperienceReplay(1000000)
        self.action_space = action_space

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

    def remember(self, state, action, reward, new_state, done):
        self.memory.update(state, action, reward, new_state, done)

    def train(self, batch_size=32, epochs=1):
        if 100 > len(self.memory.memory):
            return
        
        for epoch in range(epochs):
            self.update_count +=1

            (states, actions, rewards, next_states, dones) = self.memory.sample(batch_size)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).unsqueeze(-1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

            q = self.dqn.forward(states).gather(-1, actions.long())
            q2 = self.dqn_target.forward(next_states).max(dim=-1, keepdim=True)[0].detach()

            target = (rewards + (1 - dones) * self.gamma * q2).to(self.device)

            loss = F.mse_loss(q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.update_count % 100 == 0:
                self.update_target()

    def update_target(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
