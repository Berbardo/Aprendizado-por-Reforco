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
        self.policy1 = nn.Linear(observation_shape, 64)
        self.policy2 = nn.Linear(64, 64)
        self.policy3 = nn.Linear(64, action_shape)
        
        self.value1 = nn.Linear(observation_shape, 64)
        self.value2 = nn.Linear(64, 64)
        self.value3 = nn.Linear(64, 1)

    def forward(self, state):
        probs = F.tanh(self.policy1(state))
        probs = F.tanh(self.policy2(probs))
        probs = F.softmax(self.policy3(probs), dim=-1)
        
        v = F.tanh(self.value1(state))
        v = F.tanh(self.value2(v))
        v = self.value3(v)

        return probs, v

class SharedPPO:
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.99, lam=0.95, entropy_coef=0.05, clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.clip = clip

        self.memory = ExperienceReplay()

        self.actorcritic = ActorCritic(observation_space.shape[0], action_space.n).to(self.device)
        self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(), lr=lr, eps=1e-5)

    def prob(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dists, _ = self.actorcritic.forward(state)
        probs = Categorical(dists)
        return probs

    def act(self, state):
        probs = self.prob(state)
        action = probs.sample()
        return action.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        probs = self.prob(state)
        action_torch = torch.LongTensor(action).to(self.device)
        log_probs = probs.log_prob(action_torch)
        self.memory.update(state, action, log_probs, reward, new_state, done)

    def compute_gae(self, values, dones, rewards):
        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)
        deltas = torch.zeros_like(rewards).to(self.device)

        returns[-1] = rewards[-1] + self.gamma * (1 - dones[-1]) * rewards[-1]
        advantages[-1] = returns[-1] - values[-1]

        for i in reversed(range(len(rewards) - 1)):
            delta = rewards[i] + self.gamma * (1 - dones[i]) * values[i+1] - values[i]
            advantages[i] = delta + self.gamma * self.lam * (1 - dones[i]) * advantages[i + 1]
            returns[i] = advantages[i] + values[i]

        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    def train(self, batch_size=64, epochs=4):
        if self.memory.length < batch_size:
            return

        (states, actions, log_probs, rewards, next_states, dones) = self.memory.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device).detach()

        _, v = self.actorcritic.forward(states)
        returns, advantages = self.compute_gae(v, dones, rewards)
        
        for _ in range(epochs):
            dists, v = self.actorcritic.forward(states)

            new_probs = Categorical(dists)
            new_logprobs = new_probs.log_prob(actions)
            entropy = new_probs.entropy().mean()
            ratios = torch.exp(new_logprobs.unsqueeze(-1) - log_probs.unsqueeze(-1).detach())

            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages.detach()

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * F.smooth_l1_loss(v, returns.detach())
            entropy_loss = - self.entropy_coef * entropy

            total_loss = policy_loss + value_loss + entropy_loss

            self.actorcritic_optimizer.zero_grad()
            total_loss.backward()
            self.actorcritic_optimizer.step()