import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.experience_replay import ExperienceReplay
from utils.experience_replay import TorchReplay
from utils.experience_replay import NumpyReplay

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
        probs = torch.tanh(self.policy1(state))
        probs = torch.tanh(self.policy2(probs))
        probs = F.softmax(self.policy3(probs), dim=-1)
        probs = Categorical(probs)
        
        v = torch.tanh(self.value1(state))
        v = torch.tanh(self.value2(v))
        v = self.value3(v)

        return probs, v

class SharedPPO:
    def __init__(self, env_num, observation_space, action_space, lr=7e-4, steps=64, gamma=0.99, lam=0.95, entropy_coef=0.005, clip=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.clip = clip
        self.steps = steps

        self.memory = NumpyReplay(steps, env_num, observation_space.shape[0], self.device)

        self.actorcritic = ActorCritic(observation_space.shape[0], action_space.n).to(self.device)
        self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(), lr=lr, eps=1e-6)
        self.target_actorcritic = ActorCritic(observation_space.shape[0], action_space.n).to(self.device)
        self.target_actorcritic.load_state_dict(self.actorcritic.state_dict())

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs, _ = self.target_actorcritic.forward(state)
        action = probs.sample()
        return action.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        state_torch = torch.FloatTensor(state).to(self.device)
        probs, _ = self.actorcritic.forward(state_torch)
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

    def compute_loss(self, states, actions, logp, advantages, returns):
        new_probs, v = self.actorcritic.forward(states)
        
        new_logprobs = new_probs.log_prob(actions)
        entropy = new_probs.entropy().mean()
        ratios = torch.exp(new_logprobs.unsqueeze(-1) - logp.unsqueeze(-1).detach())

        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages.detach()

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * F.mse_loss(v, returns.detach())
        entropy_loss = - self.entropy_coef * entropy

        return policy_loss, value_loss, entropy_loss

    def train(self, epochs=8):
        if self.memory.length < self.steps:
            return

        states, actions, log_probs, rewards, next_states, dones = self.memory.sample()

        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        log_probs = log_probs.detach()
    
        _, v = self.actorcritic.forward(states)
        returns, advantages = self.compute_gae(v, dones, rewards)

        for _ in range(epochs):
            self.actorcritic_optimizer.zero_grad()

            policy_loss, value_loss, entropy_loss = self.compute_loss(states, actions, log_probs, advantages, returns)            
            total_loss = policy_loss + value_loss + entropy_loss

            total_loss.backward()
            self.actorcritic_optimizer.step()

        self.target_actorcritic.load_state_dict(self.actorcritic.state_dict())