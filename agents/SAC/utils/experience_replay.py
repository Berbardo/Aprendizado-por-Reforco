import random
from collections import deque

class ExperienceReplay:

    def __init__(self, length):
        self.reset(length)

    def reset(self, length):
        self.memory = deque(maxlen=length)
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