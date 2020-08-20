import torch
import numpy as np

class NumpyReplay:
    def __init__(self, max_length, observation_space, device):
        self.device = device
        self.index, self.size, self.max_length = 0, 0, max_length

        self.states = np.zeros((max_length, observation_space), dtype=np.float32)
        self.actions = np.zeros((max_length), dtype=np.float32)
        self.rewards = np.zeros((max_length), dtype=np.float32)
        self.next_states = np.zeros((max_length, observation_space), dtype=np.float32)
        self.dones = np.zeros((max_length), dtype=np.float32)

    def update(self, states, actions, rewards, next_states, dones):
        self.states[self.index] = states
        self.actions[self.index] = actions
        self.rewards[self.index] = rewards
        self.next_states[self.index] = next_states
        self.dones[self.index] = dones
        self.index = (self.index + 1) % self.max_length
        if self.size < self.max_length:
            self.size = self.index
            
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (torch.as_tensor(self.states[idxs]).to(self.device), torch.as_tensor(self.actions[idxs]).to(self.device),
                torch.as_tensor(self.rewards[idxs]).to(self.device), torch.as_tensor(self.next_states[idxs]).to(self.device),
                torch.as_tensor(self.dones[idxs]).to(self.device))
