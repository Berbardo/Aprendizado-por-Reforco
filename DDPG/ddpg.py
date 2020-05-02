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
        self.linear1 = nn.Linear(env.observation_space.shape[0], 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, env.action_space.shape[0])
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(env.observation_space.shape[0], 512)
        self.linear2 = nn.Linear(512 + env.action_space.shape[0], 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

    def forward(self, state, a):
        x = F.relu(self.linear1(state))
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        q = self.linear4(xa)

        return q

class DDPG:
    def __init__(self, env, alpha=0.001, beta=0.001, gamma=0.99, tau=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env  = env
        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low

        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=100000)
        self.noise = OUNoise(self.action_shape, 0)

        self.actor = Actor(self.env).to(self.device)
        self.target_actor = Actor(self.env).to(self.device)

        self.critic = Critic(self.env).to(self.device)
        self.target_critic = Critic(self.env).to(self.device)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=beta)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()
        action += self.noise.sample()

        return np.clip(action, self.action_low, self.action_high)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((np.array(state), action, reward, new_state, done))

    def train(self, batch_size=128, epochs=1):
        if batch_size > len(self.memory):
            return
        
        for epoch in range(epochs):
            minibatch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for (s, a, r, s2, d) in minibatch:
                states.append(s)
                actions.append(a)
                rewards.append([r])
                next_states.append(s2)
                dones.append(d)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            self._train_critic(states, actions, rewards, next_states, dones)
            self._train_actor(states, actions, rewards, next_states, dones)
            self.update_target()

    def _train_critic(self, states, actions, rewards, next_states, dones):
        q = self.critic.forward(states, actions)
        a2 = self.target_actor.forward(next_states)
        q2 = self.target_critic.forward(next_states, a2.detach())
        
        target = rewards + self.gamma * q2
        q_loss = F.mse_loss(q, target.detach())

        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        self.critic_optimizer.step()

    def _train_actor(self, states, actions, rewards, next_states, dones):
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def update_target(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
