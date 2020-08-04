import torch
import numpy as np

class ExperienceReplay:

    def __init__(self, length, observation_space, device):
        self.device = device
        self.states = torch.ones([length, observation_space], dtype=torch.float32, device=self.device)
        self.actions = torch.ones([length], dtype=torch.int32, device=self.device)
        self.rewards = torch.ones([length], dtype=torch.float32, device=self.device)
        self.next_states = torch.ones([length, observation_space], dtype=torch.float32, device=self.device)
        self.dones = torch.ones([length], dtype=torch.float32, device=self.device)
        self.reset(length)

    def reset(self, length):
        self.size = 0

    def update(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            self.states[self.size + i] = torch.FloatTensor(states[i]).to(self.device)
            self.actions[self.size + i] = actions[i]
            self.rewards[self.size + i] = rewards[i]
            self.next_states[self.size + i] = torch.FloatTensor(next_states[i]).to(self.device)
            self.dones[self.size + i] = int(dones[i] == True)
        self.size += len(states)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs])