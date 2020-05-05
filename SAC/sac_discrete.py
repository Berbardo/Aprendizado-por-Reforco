import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from torch.distributions import Categorical

class ExperienceReplay:

    def __init__(self):
        self.reset()

    def reset(self):
        self.memory = deque(maxlen=1000000)
        self.length = 0

    def update(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            experience = (states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.memory.append(experience)

    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        batch = random.sample(self.memory, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (states, actions, rewards, next_states, dones)

class Actor(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(observation_shape, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_shape)
        
    def forward(self, state):
        probs = F.relu(self.linear1(state))
        probs = F.relu(self.linear2(probs))
        probs = F.softmax(self.linear3(probs), dim=-1)

        dists = Categorical(probs)
        action = dists.sample()

        return action, probs, torch.log(probs + 1e-10)


class Critic(nn.Module):
    def __init__(self, observation_shape, action_shape, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(observation_shape, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_shape)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        q = F.relu(self.linear1(state))
        q = F.relu(self.linear2(q))
        q = self.linear3(q)

        return q

class SAC:
    def __init__(self, observation_space, action_space, gamma=0.99, tau=0.01, p_lr=3e-4, q_lr=3e-4, a_lr=3e-4, policy_freq=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_shape = observation_space.shape[0]
        self.action_shape = action_space.n

        self.gamma = gamma
        self.tau = tau
        self.memory = ExperienceReplay()
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

        self.target_entropy = np.log(self.action_shape) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

    def act(self, state, test=False):
        state = torch.FloatTensor(state).to(self.device)
        action, probs, _ = self.actor.forward(state)

        if test:
            return torch.argmax(probs).unsqueeze(0).cpu().detach().numpy()
        return action.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.update(state, action, reward, new_state, done)

    def train(self, batch_size=32, start_step=1000):
        if start_step > len(self.memory.memory):
            return

        self.count +=1
        
        (states, actions, rewards, next_states, dones) = self.memory.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        self._train_critics(states, actions, rewards, next_states, dones)
        if self.count % self.policy_freq == 0:
            self._train_actor(states, actions, rewards, next_states, dones)
            self.update_target()

    def _train_critics(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_probs, next_logp = self.actor.forward(next_states)

            target_Q1 = self.target_critic1.forward(next_states)
            target_Q2 = self.target_critic2.forward(next_states)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_logp
            target_Q = target_Q.gather(-1, next_actions.unsqueeze(-1).long())
            target_Q = rewards + ((1-dones) * self.gamma * target_Q).detach()
        
        current_Q1 = self.critic1(states).gather(-1, actions.long())
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.critic_optimizer1.zero_grad()
        loss_Q1.backward()
        self.critic_optimizer1.step()
        
        current_Q2 = self.critic2(states).gather(-1, actions.long())
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer2.zero_grad()
        loss_Q2.backward()
        self.critic_optimizer2.step()

    def _train_actor(self, states, actions, rewards, next_states, dones):
        new_actions, probs, logp = self.actor.forward(states)

        min_q = torch.min(self.critic1.forward(states), self.critic2.forward(states))

        policy_loss = (probs * (self.alpha * logp - min_q)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        ent = -torch.sum(logp * probs, dim=-1)
        alpha_loss = -(self.log_alpha * (self.target_entropy - ent).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

    def update_target(self):       
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))