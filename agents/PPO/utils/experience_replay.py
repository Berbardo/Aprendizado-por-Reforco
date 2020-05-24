import numpy as np
import torch
import scipy.signal

class ExperienceReplay:

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.length = 0

    def update(self, states, actions, log_probs, rewards, next_states, dones):
        self.states.append(states)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.dones.append(dones)
        self.length += 1

    def sample(self):
        states = self.states
        actions = self.actions
        log_probs = self.log_probs
        rewards = self.rewards
        next_states = self.next_states
        dones = self.dones
        self.reset()
        return states, actions, log_probs, rewards, next_states, dones

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, n_envs, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, (n_envs, obs_dim)), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, n_envs), dtype=np.float32)
        self.adv_buf = np.zeros((size, n_envs), dtype=np.float32)
        self.rew_buf = np.zeros((size, n_envs), dtype=np.float32)
        self.ret_buf = np.zeros((size, n_envs), dtype=np.float32)
        self.val_buf = np.zeros((size, n_envs), dtype=np.float32)
        self.logp_buf = np.zeros((size, n_envs), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def update(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        rews = np.append(self.rew_buf, last_val)
        vals = np.append(self.val_buf, last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        obs = torch.FloatTensor(self.obs_buf)
        act = torch.FloatTensor(self.act_buf)
        logp = torch.FloatTensor(self.logp_buf)
        ret = torch.FloatTensor(self.ret_buf)
        adv = torch.FloatTensor(self.adv_buf)
        return obs, act, logp, ret, adv