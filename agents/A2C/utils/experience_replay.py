import numpy as np

class ExperienceReplay:

    def __init__(self, max_length, observation_space):
        self.length = 0
        self.max_length = max_length

        self.states = np.zeros((max_length, observation_space), dtype=np.float32)
        self.actions = np.zeros((max_length), dtype=np.int32)
        self.rewards = np.zeros((max_length), dtype=np.float32)
        self.next_states = np.zeros((max_length, observation_space), dtype=np.float32)
        self.dones = np.zeros((max_length), dtype=np.float32)

    def update(self, states, actions, rewards, next_states, dones):
        self.states[self.length] = states
        self.actions[self.length] = actions
        self.rewards[self.length] = rewards
        self.next_states[self.length] = next_states
        self.dones[self.length] = dones
        self.length += 1

    def sample(self):
        self.length = 0

        return (self.states, self.actions, self.rewards, self.next_states, self.dones)

