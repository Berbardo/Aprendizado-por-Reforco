import numpy as np
from collections import deque

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim, n_envs, size: int, n_step: int = 1, gamma: float = 0.99):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = [deque(maxlen=n_step) for i in range(n_envs)]
        self.n_step = n_step
        self.gamma = gamma

    def store(self, obs, act, rew, next_obs, done, i):
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer[i].append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer[i]) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer[i], self.gamma
        )
        obs, act = self.n_step_buffer[i][0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[i][0]

    # def sample_batch(self):
    #     idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
    #     batch = np.array(self.buffer)[idxs]

    #     return batch, idxs
    
    def sample_batch_from_idxs(self, idxs):
        # for N-step Learning
        states=self.obs_buf[idxs]
        actions=self.acts_buf[idxs]
        rewards=self.rews_buf[idxs]
        next_states=self.next_obs_buf[idxs]
        dones=self.done_buf[idxs]

        return (states, actions, rewards, next_states, dones)
    
    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size