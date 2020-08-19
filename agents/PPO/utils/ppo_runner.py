import time
import math
import numpy as np

def train(agent, env, total_timesteps, break_condition):
    total_reward = 0
    episode_returns = []
    avg_total_rewards = []

    observation = env.reset()
    timestep = 0
    episode = 0

    start_time = time.time()

    while timestep < total_timesteps:
        action = agent.act(observation)
        next_observation, reward, done, _ = env.step(action)
        agent.remember(observation, action, reward, next_observation, done)
        agent.train()
        
        timestep += 1

        total_reward += reward

        if done:
            episode_returns.append(total_reward)
            episode += 1

        if any(G for G in episode_returns):
            avg_total_rewards.append(np.mean([G[-1] for G in episode_returns[-20:]]))

        total_reward *= 1 - done
        observation = next_observation

        ratio = math.ceil(100 * timestep / total_timesteps)
        uptime = math.ceil(time.time() - start_time)

        avg_return = avg_total_rewards[-1] if avg_total_rewards else np.nan

        print(f"[{ratio:3d}% / {uptime:3d}s] timestep = {timestep}/{total_timesteps}, episode = {episode:3d}, avg_return = {avg_return:10.4f}\r", end="")

        if avg_return > break_condition:
            return avg_total_rewards

    return avg_total_rewards

def vector_train(agent, env, total_timesteps, break_condition):
    total_rewards = [[] for _ in range(env.num_envs)]
    avg_total_rewards = []

    total_reward = np.zeros(env.num_envs)
    observations = env.reset()
    timestep = 0
    episode = 0

    t = 0

    start_time = time.time()

    while timestep < total_timesteps:
        actions = agent.act(observations)
        next_observations, rewards, dones, _ = env.step(actions)
        agent.remember(observations, actions, rewards, next_observations, dones)
        agent.train()
        
        timestep += len(observations)
        t += 1

        total_reward += rewards

        for i in range(env.num_envs):
            if dones[i]:
                total_rewards[i].append((t, timestep, total_reward[i]))
                episode += 1

        if any(G for G in total_rewards):
            episode_returns = sorted(
                list(np.concatenate([G for G in total_rewards if G])),
                key=lambda x: x[1]
            )

            avg_total_rewards.append(np.mean([G[-1] for G in episode_returns[-20:]]))

        total_reward *= 1 - dones
        observations = next_observations

        ratio = math.ceil(100 * timestep / total_timesteps)
        uptime = math.ceil(time.time() - start_time)

        avg_return = avg_total_rewards[-1] if avg_total_rewards else np.nan

        print(f"[{ratio:3d}% / {uptime:3d}s] timestep = {timestep}/{total_timesteps}, episode = {episode:3d}, avg_return = {avg_return:10.4f}\r", end="")

        if avg_return > break_condition:
            print("\n")
            return avg_total_rewards

    print("\n")
    return avg_total_rewards

def evaluate(agent, env, n_episodes=5, render=False):

    for episode in range(n_episodes):

        obs = env.reset()        
        total_reward = 0.0
        episode_length = 0

        done = False
        while not done:
            action = agent.act(obs[None,:])[0]
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            
            total_reward += reward
            episode_length += 1

            if render:
                env.render()

        print(f">> episode = {episode} / {n_episodes}, total_reward = {total_reward:10.4f}, episode_length = {episode_length}")
        
    if render:
        env.close()