import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def pg_loss(y_actual, y_pred):
    actions = tf.cast(y_actual[:, 0], tf.int32)
    td = y_actual[:, 1]
    selector = tf.stack([tf.range(tf.size(actions)), actions], axis=1)
    logp = tf.math.log(tf.gather_nd(y_pred, selector) + 10e-10)
    return -tf.math.reduce_sum(logp * td)

class Actor:
    def __init__(self, env, lr=0.001):
        self.env = env
        self.lr = lr
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.model = self.create_model()
        
    def create_model(self):
        model = Sequential([
            Dense(32, input_shape=self.state_shape, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='softmax'),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=pg_loss)
        return model
    
    def act(self, state):
        state = state.reshape(1, *state.shape)
        policy = self.model.predict(state).flatten()
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

class Critic:
    def __init__(self, env, lr=0.001):
        self.env = env
        self.lr = lr
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.model = self.create_model()
        
    def create_model(self):
        model = Sequential([
            Dense(32, input_shape=self.state_shape, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear'),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss="mse")
        return model
        
    def v(self, state):
        return self.model.predict(state.reshape(1, *state.shape))[0]

class TDActorCritic:
    def __init__(self, env, lr=0.001, epsilon=.99995, gamma=0.99):
        self.env  = env
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.gamma = gamma

        self.actor = Actor(self.env, self.lr)
        self.critic = Critic(self.env, self.lr)

    def train(self, state, action, reward, new_state, done):
        v = self.critic.v(state)
        
        if done:
            target = -reward
        else:
            v2 = self.critic.v(new_state)[0]
            target = reward + self.gamma * v2

        td_error = target - v
        target = [target]
        state = state.reshape(1, *state.shape)
        
        self.critic.model.train_on_batch(state, target)
        
        y_actual = np.empty(shape=(1, 2))
        y_actual[:, 0] = action
        y_actual[:, 1] = td_error
        loss = self.actor.model.train_on_batch(state, y_actual)

    def act(self, state):
        self.epsilon *= self.epsilon
        self.epsilon = max(self.epsilon, self.epsilon_min)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor.act(state)