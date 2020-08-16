import gym

from dqn import DQN
from ddqn import DDQN
from dueling_ddqn import DuelingDDQN
from noisy_dqn import NoisyDQN
from categorical_dqn import CategoricalDQN
from rainbow import Rainbow

from utils.dqn_runner import vector_train
from utils.dqn_runner import evaluate

if __name__ == "__main__":
    env = gym.vector.make("CartPole-v1", num_envs=4, asynchronous=True)
    agent = DDQN(env.single_observation_space, env.single_action_space)

    returns = vector_train(agent, env, 50000, 450)

    eval_env = gym.make("CartPole-v1")
    evaluate(agent, eval_env, 1, True)