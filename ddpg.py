import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from ou_noise import OUNoise
from ou_noise import OrnsteinUhlenbeckActionNoise

def custom_loss(y_pred, y_actual):
    policy_loss = -tf.reduce_mean(y_actual)
    return policy_loss

class Actor:
    def __init__(self, env, lr=0.0001):
        self.env = env
        self.lr = lr
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.model = self.create_model()
        
    def create_model(self):
        input_layer = Input(shape=self.state_shape)
        dense1 = Dense(512, activation='relu')(input_layer)
        dense2 = Dense(128, activation='relu')(dense1)
        action = Dense(self.action_shape[0], activation='tanh')(dense2)
        actor = Model(inputs= input_layer, outputs= action)
        actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=custom_loss)
        return actor
    
    def act(self, state):
        state = state.reshape(1, *state.shape)
        action = self.model.predict(state).flatten()
        return action

class Critic:
    def __init__(self, env, lr=0.001):
        self.env = env
        self.lr = lr
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.model = self.create_model()
        
    def create_model(self):
        state_input = Input(shape=self.state_shape)
        action_input = Input(shape=self.action_shape)
        
        state1 = Dense(500, activation='relu')(state_input)
        merged = Concatenate()([state1, action_input])
        dense2 = Dense(300, activation='relu')(merged)

        q = Dense(1, activation='linear')(dense2)

        critic = Model(inputs=[state_input,action_input], outputs= q)
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss="mse")
        return critic
        
    def q(self, state, action):
        return self.model.predict([state.reshape(1, *state.shape), action.reshape(1, *action.shape)])[0]

class DDPG:
    def __init__(self, env, alpha=0.0005, beta=0.0005, gamma=0.99, tau=0.001):
        self.env  = env
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low

        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=100000)
        self.noise = OrnsteinUhlenbeckActionNoise(self.env.action_space)

        self.actor = Actor(self.env, alpha)
        self.target_actor = Actor(self.env, alpha)
        self.critic = Critic(self.env, beta)
        self.target_critic = Critic(self.env, beta)

        critic_weights = self.critic.model.get_weights()
        self.target_critic.model.set_weights(critic_weights)
        actor_weights = self.actor.model.get_weights()
        self.target_actor.model.set_weights(actor_weights)

    def act(self, state, step):
        action = 2*self.actor.act(state)
        action += self.noise.get_noise()

        return np.clip(action, self.action_low, self.action_high)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def _train_critic(self, batch):
        states = []
        actions = []
        target = []
        for (s, a, r, s2, done) in batch:
            if done:
                t = r

            else:
                a2 = self.target_actor.act(s2)
                q2 = self.target_critic.q(s2, a2)
                t = r + self.gamma * q2

            states.append(s)
            actions.append(a)
            target.append(t)

        states = np.array(states)
        actions = np.array(actions)
        target = np.array(target)

        self.critic.model.fit([states, actions], target, steps_per_epoch=1, verbose=0)

    def _train_actor(self, batch):
        states = []
        actions = []
        target = []

        for (s, a, r, s2, done) in batch:
            action = self.actor.act(s)

            states.append(s)
            actions.append(action)

        states = np.array(states)
        actions = np.array(actions)
        q = self.critic.model.predict([states, actions]).flatten()
        q = np.array(q)

        self.actor.model.fit(states, q, steps_per_epoch=1, verbose=0)

    def train(self, batch_size=128, epochs=50):
        if batch_size > len(self.memory):
            return
        
        for epoch in range(epochs):
            minibatch = random.sample(self.memory, batch_size)
            self._train_critic(minibatch)
            self._train_actor(minibatch)
            self.update_target()

    def _update_target_actor(self):
        actor_weights  = self.actor.model.get_weights()
        target_actor_weights = self.target_actor.model.get_weights()

        for i in range(len(target_actor_weights)):
            target_actor_weights[i] = actor_weights[i]*self.tau + target_actor_weights[i]*(1-self.tau)
        self.target_actor.model.set_weights(target_actor_weights)

    def _update_target_critic(self):
        critic_weights  = self.critic.model.get_weights()
        target_critic_weights = self.target_critic.model.get_weights()

        for i in range(len(target_critic_weights)):
            target_critic_weights[i] = critic_weights[i]*self.tau + target_critic_weights[i]*(1-self.tau)
        self.target_critic.model.set_weights(target_critic_weights)

    def update_target(self):
        self._update_target_actor()
        self._update_target_critic()
