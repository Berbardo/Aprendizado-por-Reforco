import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.experience_replay import ExperienceReplay

class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(ActorCritic, self).__init__()
        self.policy1 = nn.Linear(observation_shape, 256)
        self.policy2 = nn.Linear(256, action_shape)
        
        self.value1 = nn.Linear(observation_shape, 256)
        self.value2 = nn.Linear(256, 1)

    def forward(self, state):
        probs = F.relu(self.policy1(state))
        probs = F.softmax(self.policy2(probs), dim=1)
        
        v = F.relu(self.value1(state))
        v = self.value2(v)

        return probs, v

class SharedA2C:
    def __init__(self, observation_space, action_space, lr=5e-4, gamma=0.99, lam=0.95, entropy_coef=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

        self.memory = ExperienceReplay()

        self.actorcritic = ActorCritic(observation_space.shape[0], action_space.n).to(self.device)
        self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(), lr=lr)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dists, _ = self.actorcritic.forward(state)
        probs = Categorical(dists)
        action = probs.sample().cpu().detach().numpy()
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.update(state, action, reward, new_state, done)

    def train(self):
        if self.memory.length < 16:
            return

        (states, actions, rewards, next_states, dones) = self.memory.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(2).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(2).to(self.device)

        dists, v = self.actorcritic.forward(states)
        _, v2 = self.actorcritic.forward(next_states)

        deltas = rewards + (1 - dones) * self.gamma * v2 - v

        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)

        returns[-1] = rewards[-1] + self.gamma * (1 - dones[-1]) * v2[-1]
        advantages[-1] = deltas[-1]

        for i in reversed(range(len(rewards) - 1)):
            returns[i] = rewards[i] + self.gamma * (1 - dones[i]) * returns[i + 1]
            advantages[i] = deltas[i] + self.gamma * self.lam * (1 - dones[i]) * advantages[i + 1]

        probs = Categorical(dists)

        logp = -probs.log_prob(actions)

        entropy = probs.entropy().mean()

        policy_loss = (logp.unsqueeze(2) * advantages.detach()).mean()
        value_loss = F.mse_loss(v, returns.detach())
        entropy_loss = - self.entropy_coef * entropy

        total_loss = policy_loss + value_loss + entropy_loss

        self.actorcritic_optimizer.zero_grad()
        total_loss.backward()
        self.actorcritic_optimizer.step()