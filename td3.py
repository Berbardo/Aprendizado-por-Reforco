import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from ou_noise import OUNoise

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(env.observation_space.shape[0], 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.linear2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.linear3 = nn.Linear(300, env.action_space.shape[0])
        
    def forward(self, state):
        a = F.relu(self.bn1(self.linear1(state)))
        a = F.relu(self.bn2(self.linear2(a)))
        a = torch.tanh(self.linear3(a))

        return a

class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.linear2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.linear3 = nn.Linear(300, 1)

    def forward(self, state, a):
        sa = torch.cat([state,a], 1)
        q = F.relu(self.bn1(self.linear1(sa)))
        q = F.relu(self.bn2(self.linear2(q)))
        q = self.linear3(q)

        return q

class TD3:
    def __init__(self, env, alpha=0.0005, beta=0.0005, gamma=0.99, tau=0.005, policy_noise=0.1, noise_clip=0.3, policy_freq=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env  = env
        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low

        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=1000000)
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.count = 0

        self.noise = OUNoise(self.action_shape, 0)

        self.actor = Actor(self.env).to(self.device)
        self.target_actor = Actor(self.env).to(self.device)

        self.critic1 = Critic(self.env).to(self.device)
        self.target_critic1 = Critic(self.env).to(self.device)
        self.critic2 = Critic(self.env).to(self.device)
        self.target_critic2 = Critic(self.env).to(self.device)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=beta)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=beta)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state)
        self.actor.train()
        action = action.squeeze(0).cpu().detach().numpy()
        action += self.noise.sample()

        return np.clip(action, self.action_low, self.action_high)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((np.array(state), action, reward, new_state, done))

    def train(self, batch_size=128):
        if batch_size > len(self.memory):
            return

        self.count +=1
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for (s, a, r, s2, d) in minibatch:
            states.append(s)
            actions.append(a)
            rewards.append([r])
            next_states.append(s2)
            dones.append([d])

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        self._train_critics(states, actions, rewards, next_states, dones)
        if self.count % self.policy_freq == 0:
            self._train_actor(states, actions, rewards, next_states, dones)
            self.update_target()

    def _train_critics(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            noise = actions.data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor.forward(next_states) + noise)
            next_actions = next_actions.clamp(-1, 1)

            target_Q1 = self.target_critic1(next_states, next_actions)
            target_Q2 = self.target_critic2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + ((1-dones) * self.gamma * target_Q).detach()
        
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
        policy_loss = -self.critic1.forward(states, self.actor.forward(states)).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def update_target(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))