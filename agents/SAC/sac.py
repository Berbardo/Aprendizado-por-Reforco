import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from utils.experience_replay import NumpyReplay

class Actor(nn.Module):
    def __init__(self, observation_shape, action_shape, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(observation_shape, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_shape)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(256, action_shape)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        a = F.relu(self.linear1(state))
        a = F.relu(self.linear2(a))

        mean = self.mean_linear(a)
        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = (normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_pi


class Critic(nn.Module):
    def __init__(self, observation_shape, action_shape, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(observation_shape + action_shape, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, a):
        sa = torch.cat([state,a], 1)
        q = F.relu(self.linear1(sa))
        q = F.relu(self.linear2(q))
        q = self.linear3(q)

        return q

class SAC:
    def __init__(self, observation_space, action_space, alpha=0.2, gamma=0.99, tau=0.01, p_lr=7e-4, q_lr=7e-4, a_lr=3e-4, policy_freq=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_shape = observation_space.shape[0]
        self.action_shape = action_space.shape[0]
        self.action_range = [action_space.low, action_space.high]


        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.memory = NumpyReplay(1000000, observation_space.shape[0], self.device)
        self.count = 0
        self.policy_freq = policy_freq

        self.actor = Actor(self.state_shape, self.action_shape).to(self.device)

        self.critic1 = Critic(self.state_shape, self.action_shape).to(self.device)
        self.target_critic1 = Critic(self.state_shape, self.action_shape).to(self.device)
        self.critic2 = Critic(self.state_shape, self.action_shape).to(self.device)
        self.target_critic2 = Critic(self.state_shape, self.action_shape).to(self.device)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=p_lr)
        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=q_lr)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=q_lr)

        self.target_entropy = -torch.prod(torch.Tensor(self.action_shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

    def act(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _ = self.actor.sample(state)
        action = action.cpu().detach().squeeze(0).numpy()

        return self.rescale_action(action)

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
            (self.action_range[1] + self.action_range[0]) / 2.0

    def remember(self, state, action, reward, new_state, done):
        for i in range(len(state)):
            self.memory.update(state[i], action[i], reward[i], new_state[i], done[i])

    def train(self, batch_size=64):
        if batch_size > self.memory.size:
            return

        self.count +=1
        
        (states, actions, rewards, next_states, dones) = self.memory.sample(batch_size)

        actions = actions.unsqueeze(-1)
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

        self._train_critics(states, actions, rewards, next_states, dones)
        if self.count % self.policy_freq == 0:
            self._train_actor(states, actions, rewards, next_states, dones)
            self.update_target()

    def _train_critics(self, states, actions, rewards, next_states, dones):
        next_actions, next_log_pi = self.actor.sample(next_states)

        with torch.no_grad():
            target_Q1 = self.target_critic1.forward(next_states, next_actions)
            target_Q2 = self.target_critic2.forward(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pi
            target_Q = rewards + ((1-dones) * self.gamma * target_Q)
        
        current_Q1 = self.critic1(states, actions)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.critic_optimizer1.zero_grad()
        loss_Q1.backward()
        self.critic_optimizer1.step()
        
        current_Q2 = self.critic2(states, actions)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer2.zero_grad()
        loss_Q2.backward()
        self.critic_optimizer2.step()

    def _train_actor(self, states, actions, rewards, next_states, dones):
        new_actions, log_pi = self.actor.sample(states)
        min_q = torch.min(self.critic1.forward(states, new_actions), self.critic2.forward(states, new_actions))

        policy_loss = (self.alpha * log_pi - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

    def update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)