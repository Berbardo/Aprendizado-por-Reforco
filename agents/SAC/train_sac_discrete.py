import gym
from sac_discrete2 import DiscreteSAC

from utils.sac_runner import vector_train
from utils.sac_runner import evaluate

if __name__ == "__main__":
    env = gym.vector.make("CartPole-v1", num_envs=8, asynchronous=True)
    actor = DiscreteSAC(env.single_observation_space, env.single_action_space)

    returns = vector_train(actor, env, 40000, 400)

    eval_env = gym.make("CartPole-v1")
    evaluate(actor, eval_env, 1, True)