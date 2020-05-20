import gym
from agentes.SAC.sac import SAC
from rl_utils.runner import vector_train
from rl_utils.runner import evaluate

if __name__ == "__main__":
    print("oi")

    env = gym.vector.make("Pendulum-v0", num_envs=4, asynchronous=True)
    actor = SAC(env.single_observation_space, env.single_action_space, p_lr=1e-3, q_lr=1e-3)

    returns = vector_train(actor, env, 40000, -200)

    eval_env = gym.make("Pendulum-v0")
    evaluate(actor, eval_env, 5, True)