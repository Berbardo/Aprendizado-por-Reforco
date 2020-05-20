from collections import deque

class ExperienceReplay:

    def __init__(self):
        self.reset()

    def reset(self):
        self.memory = deque(maxlen=1000)
        self.length = 0

    def update(self, states, actions, rewards, next_states, dones):
        experience = (states, actions, rewards, next_states, dones)
        self.length += 1
        self.memory.append(experience)

    def sample(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for experience in self.memory:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        self.reset()
        return (states, actions, rewards, next_states, dones)
