import numpy as np
from collections import deque

class ExperienceReplay:

    def __init__(self, length):
        self.reset(length)

    def reset(self, length):
        self.size = 0
        self.states = deque(maxlen=length)
        self.actions = deque(maxlen=length)
        self.rewards = deque(maxlen=length)
        self.next_states = deque(maxlen=length)
        self.dones = deque(maxlen=length)

    def update(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            self.states.append(states[i])
            self.actions.append(actions[i])
            self.rewards.append(rewards[i])
            self.next_states.append(next_states[i])
            self.dones.append(dones[i])
        self.size = len(self.states)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        next_states = np.array(self.next_states)
        dones = np.array(self.dones)

        return (states[idxs], actions[idxs], rewards[idxs], next_states[idxs], dones[idxs])