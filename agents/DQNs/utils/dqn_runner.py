import time
import math
import numpy as np
from collections import deque

def train(agent, env, total_timesteps):
    total_reward = 0
    episode_returns = deque(maxlen=20)
    avg_returns = []

    state = env.reset()
    timestep = 0
    episode = 0

    while timestep < total_timesteps:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        
        timestep += 1

        total_reward += reward


        if done:
            episode_returns.append(total_reward)
            episode += 1
            next_state = env.reset()

        if any(G for G in episode_returns):
            avg_returns.append(np.mean(episode_returns))

        total_reward *= 1 - done
        state = next_state

        ratio = math.ceil(100 * timestep / total_timesteps)

        avg_return = avg_returns[-1] if avg_returns else np.nan
        
        print(f"\r[{ratio:3d}%] timestep = {timestep}/{total_timesteps}, episode = {episode:3d}, avg_return = {avg_return:10.4f}", end="")

    return avg_returns

def evaluate(agent, env, episodes=10, render=False):
    total_reward = 0
    episode_returns = deque(maxlen=episodes)
    
    episode = 0

    state = env.reset()

    while episode < episodes:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)        
       
        total_reward += reward

        if render:
            env.render()

        if done:
            episode_returns.append(total_reward)
            episode += 1
            next_state = env.reset()

        total_reward *= 1 - done
        state = next_state

        ratio = math.ceil(100 * episode / episodes)
        
        print(f"\r[{ratio:3d}%] episode = {episode:3d}, avg_return = {np.mean(episode_returns):10.4f}", end="")

    env.close()

    return np.mean(episode_returns)