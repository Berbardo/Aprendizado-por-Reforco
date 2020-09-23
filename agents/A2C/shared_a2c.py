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
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.99, lam=0.95, entropy_coef=0.001, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

        self.batch_size = batch_size
        self.memory = ExperienceReplay(batch_size, observation_space.shape[0])

        self.actorcritic = ActorCritic(observation_space.shape[0], action_space.n)
        self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(), lr=lr)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        dists, _ = self.actorcritic.forward(state)
        probs = Categorical(dists)
        action = probs.sample().cpu().detach().numpy()[0]
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.update(state, action, reward, new_state, done)

    def compute_gae(self, rewards, dones, v, v2):
        T = len(rewards)

        returns = torch.zeros_like(rewards)
        gaes = torch.zeros_like(rewards)
        
        future_gae = torch.tensor(0.0, dtype=rewards.dtype)
        next_return = torch.tensor(v2[-1], dtype=rewards.dtype)

        not_dones = 1 - dones
        deltas = rewards + not_dones * self.gamma * v2 - v

        for t in reversed(range(T)):
            returns[t] = next_return = rewards[t] + self.gamma * not_dones[t] * next_return
            gaes[t] = future_gae = deltas[t] + self.gamma * self.lam * not_dones[t] * future_gae

        return gaes, returns

    def train(self):
        if self.memory.length < self.batch_size:
            return

        (states, actions, rewards, next_states, dones) = self.memory.sample()

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)

        dists, v = self.actorcritic.forward(states)
        with torch.no_grad():
            _, v2 = self.actorcritic.forward(next_states)

        advantages, returns = self.compute_gae(rewards, dones, v, v2)

        probs = Categorical(dists)

        logp = -probs.log_prob(actions)

        entropy = probs.entropy().mean()

        policy_loss = (logp.unsqueeze(-1) * advantages.detach()).mean()
        value_loss = F.mse_loss(v, returns.detach())
        entropy_loss = - self.entropy_coef * entropy

        total_loss = policy_loss + value_loss + entropy_loss

        self.actorcritic_optimizer.zero_grad()
        total_loss.backward()
        self.actorcritic_optimizer.step()