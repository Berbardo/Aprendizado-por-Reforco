import gym
import matplotlib.pyplot as plt

from dqn import DQN
from ddqn import DDQN
# from dueling_ddqn import DuelingDDQN
# from noisy_dqn import NoisyDQN
# from categorical_dqn import CategoricalDQN
# from rainbow import Rainbow

from utils.dqn_runner import train
from utils.dqn_runner import evaluate

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = DDQN(env.observation_space, env.action_space)

    returns = train(agent, env, 25000)

    eval_env = gym.make("CartPole-v1")
    evaluate(agent, eval_env, 10, True)

    plt.plot(returns, 'r')
    plt.show(block=True)