import gym
from ppo import PPO
from shared_ppo import SharedPPO

from utils.ppo_runner import vector_train
from utils.ppo_runner import evaluate

if __name__ == "__main__":
    env = gym.vector.make("CartPole-v1", num_envs=8, asynchronous=True)
    actor = SharedPPO(8, env.single_observation_space, env.single_action_space)

    returns = vector_train(actor, env, 100000, 450)

    eval_env = gym.make("CartPole-v1")
    evaluate(actor, eval_env, 1, True)