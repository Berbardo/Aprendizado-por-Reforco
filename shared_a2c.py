import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def pg_loss(y_actual, y_pred):
    actions = tf.cast(y_actual[:, 0], tf.int32)
    adv = y_actual[:, 1]
    selector = tf.stack([tf.range(tf.size(actions)), actions], axis=1)
    logp = tf.math.log(tf.gather_nd(y_pred, selector) + 10e-10)
    return -tf.math.reduce_sum(logp * adv)

class SharedA2C:
    def __init__(self, env, epsilon=.99995, gamma=0.99):
        self.env  = env
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.epsilon = epsilon
        self.epsilon_min = 0.001
        self.gamma = gamma
        self.memory = deque(maxlen=20000)

        self.actor, self.critic = self.create_model()

    def create_model(self):
        input_layer = Input(shape=self.state_shape)
        dense1 = Dense(128, activation='relu')(input_layer)
        dense2 = Dense(128, activation='relu')(dense1)
        probs = Dense(self.action_size, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        actor = Model(inputs= input_layer, outputs=probs)
        actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=pg_loss)

        critic = Model(inputs= input_layer, outputs=values)
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

        return actor, critic

    def v(self, state):
        return self.critic.predict(state.reshape(1, *state.shape))[0]

    def get_action(self, state):
        state = state.reshape(1, *state.shape)
        policy = self.actor.predict(state).flatten()
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train(self, batch_size=32):        
        if batch_size >= len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, k=batch_size)

        states = np.array([mem[0] for mem in minibatch])
        actions = []
        y = []
        adv = []
        for i, (s, a, r, s2, done) in enumerate(minibatch):
            v = self.v(s)
            if done:
                target = -10
            else:
                v2 = self.v(s2)[0]
                target = r + self.gamma * v2
                
            actions.append(a)
            y.append(target)
            adv.append(target - v)
        
        self.critic.fit(states, y, batch_size=batch_size, verbose=0)
        
        y_actual = np.empty(shape=(len(states), 2))
        y_actual[:, 0] = actions
        y_actual[:, 1] = adv
        loss = self.actor.fit(states, y_actual, batch_size=batch_size, verbose=0)

    def act(self, state):
        self.epsilon *= self.epsilon
        self.epsilon = max(self.epsilon, self.epsilon_min)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.get_action(state)