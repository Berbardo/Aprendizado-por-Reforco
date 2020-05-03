# Implementações de Aprendizado por Reforço



![Exemplo](/img/cartpole.gif)

## A2C

Advantage Actor Critic com Generalized Advantage Estimator

### CartPole

Após 10000 timesteps

![A2C](img/CartPoleA2C.gif)

Curva de Aprendizado:

![A2C](img/CartPoleA2C.png)

## Shared Network AAC

Advantage Actor Critic com uma rede neural compartilhada entre o Actor e o Critic.

Curva de Aprendizado:

![Shared A2C](img/SharedA2C.png)

Shared AAC após 100 episódios:

![Shared A2C](img/SharedA2C.gif)

## PPO

Proximal Policy Optimization com GAE

![PPO](img/PPO.gif)

## Shared Network PPO

Shared Network Proximal Policy Optimization com GAE

Curva de Aprendizado:

![PPO](img/PPO.png)

## Soft Actor Critic

Soft Actor Critic

BipedalWalker-v2 após 170 episódios

![SAC](img/BipedalSAC.gif)

Pendulum-v0:

![SAC](img/PendulumSAC.gif)

Curva de Aprendizado:

![SAC](img/BipedalSAC.png)