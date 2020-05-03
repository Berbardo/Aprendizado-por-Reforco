import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ExperienceReplay:

    def __init__(self):
        self.reset()

    def reset(self):
        self.memory = deque(maxlen=1000)
        self.length = 0

    def update(self, states, actions, rewards, next_states, dones):
        experience = (states, actions, rewards, next_states, dones)
        self.length += 1
        self.memory.append(experience)

    def sample(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for experience in self.memory:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        self.reset()
        return (states, actions, rewards, next_states, dones)

class Actor(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(observation_shape, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, action_shape)

    def forward(self, state):
        dists = F.relu(self.linear1(state))
        dists = F.relu(self.linear2(dists))
        dists = F.softmax(self.linear3(dists), dim=1)

        return dists

class Critic(nn.Module):
    def __init__(self, observation_shape):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(observation_shape, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        v = F.relu(self.linear1(state))
        v = F.relu(self.linear2(v))
        v = self.linear3(v)

        return v

class A2C:
    def __init__(self, observation_space, action_space, p_lr=5e-4, v_lr=2e-3, gamma=0.99, lam=0.95, entropy_coef=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

        self.memory = ExperienceReplay()

        self.actor = Actor(observation_space.shape[0], action_space.n).to(self.device)
        self.critic = Critic(observation_space.shape[0]).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=p_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=v_lr)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dists = self.actor.forward(state)
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

        v = self.critic.forward(states)
        v2 = self.critic.forward(next_states)

        deltas = rewards + (1 - dones) * self.gamma * v2 - v

        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)

        returns[-1] = rewards[-1] + self.gamma * (1 - dones[-1]) * v2[-1]
        advantages[-1] = deltas[-1]

        for i in reversed(range(len(rewards) - 1)):
            returns[i] = rewards[i] + self.gamma * (1 - dones[i]) * returns[i + 1]
            advantages[i] = deltas[i] + self.gamma * self.lam * (1 - dones[i]) * advantages[i + 1]

        dists = self.actor.forward(states)
        probs = Categorical(dists)

        logp = -probs.log_prob(actions)

        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()

        policy_loss = (logp.unsqueeze(2) * advantages.detach()).mean()
        value_loss = F.mse_loss(v, returns.detach())
        entropy_loss = self.entropy_coef * (entropy).mean()

        self.actor_optimizer.zero_grad()
        (policy_loss + entropy_loss).backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.trajectory = []

