import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(observation_shape, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, action_shape)

    def forward(self, state):
        a = F.relu(self.linear1(state))
        a = F.relu(self.linear2(a))
        a = torch.softmax(self.linear3(a))

        return a

class Critic(nn.Module):
    def __init__(self, observation_shape):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(observation_shape, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, state):
        v = F.relu(self.linear1(state))
        v = F.relu(self.linear2(v))
        v = self.linear3(v)

        return v

class A2C:
    def __init__(self, observation_space, action_space, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epsilon = 0.7
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.001
        self.gamma = gamma

        self.trajectory = []

        self.actor = Actor(observation_space.shape[0], action_space.n)
        self.critic = Critic(observation_space.shape[0])

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor.forward(state)

    def remember(self, state, action, reward, new_state, done):
        for env in range(len(state)):
            self.trajectory.append([state[env], action[env], reward[env], new_state[env], done[env]])

    def train(self):
        states, actions, rewards, new_states, dones = self.trajectory

        self.trajectory = []

