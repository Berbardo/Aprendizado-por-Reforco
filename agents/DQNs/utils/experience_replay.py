import torch
import numpy as np

class ExperienceReplay:
    def __init__(self, max_length, observation_space, device):
        self.device = device
        self.index = 0
        self.size = 0
        self.max_length = max_length
        self.states = torch.ones([max_length, observation_space], dtype=torch.float32, device=self.device)
        self.actions = torch.ones([max_length], dtype=torch.int32, device=self.device)
        self.rewards = torch.ones([max_length], dtype=torch.float32, device=self.device)
        self.next_states = torch.ones([max_length, observation_space], dtype=torch.float32, device=self.device)
        self.dones = torch.ones([max_length], dtype=torch.float32, device=self.device)

    def update(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            self.states[self.index + i] = torch.FloatTensor(states[i]).to(self.device)
            self.actions[self.index + i] = actions[i]
            self.rewards[self.index + i] = rewards[i]
            self.next_states[self.index + i] = torch.FloatTensor(next_states[i]).to(self.device)
            self.dones[self.index + i] = int(dones[i] == True)
        self.index += len(states)
        if self.size < self.max_length:
            self.size = self.index
        if self.index >= self.max_length:
            self.index = 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs])