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

class ActorCritic:
    def __init__(self, env, lr=0.001, epsilon=.99995, gamma=0.99):
        self.env  = env
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.gamma = gamma
        self.memory = deque(maxlen=2000)

        self.actor = Actor(self.env, self.lr)
        self.critic = Critic(self.env, self.lr)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train(self):
        batch_size = 16
        
        if batch_size >= len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, k=batch_size)

        states = np.array([mem[0] for mem in minibatch])
        actions = []
        y = []
        td_error = []
        for i, (s, a, r, s2, done) in enumerate(minibatch):
            v = self.critic.v(s)
            if done:
                target = -r
            else:
                v2 = self.critic.v(s2)[0]
                target = r + self.gamma * v2
                
            actions.append(a)
            y.append(target)
            td_error.append(target - v)
        
        self.critic.model.train_on_batch(states, y)
        
        y_actual = np.empty(shape=(len(states), 2))
        y_actual[:, 0] = actions
        y_actual[:, 1] = td_error
        loss = self.actor.model.train_on_batch(states, y_actual)

    def act(self, state):
        self.epsilon *= self.epsilon
        self.epsilon = max(self.epsilon, self.epsilon_min)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor.act(state)