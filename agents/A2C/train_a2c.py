import gym
from a2c import A2C
from shared_a2c import SharedA2C

from utils.a2c_runner import vector_train, train
from utils.a2c_runner import evaluate

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    actor = SharedA2C(env.observation_space, env.action_space)

    returns = train(actor, env, 100000, 450)

    eval_env = gym.make("CartPole-v1")
    evaluate(actor, eval_env, 1, True)