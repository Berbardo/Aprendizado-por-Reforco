import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

clip = 0.2
entropy = 0.005

def ppo_loss(y_actual, y_pred):
    actions = tf.cast(y_actual[:, 0], tf.int32)
    selector = tf.stack([tf.range(tf.size(actions)), actions], axis=1)
    adv = y_actual[:, 1]
    probs = tf.cast(y_actual[:, 2:], tf.float32)
    
    prob = tf.gather_nd(y_pred, selector)
    old_prob = tf.gather_nd(probs, selector)
    ratio = prob/(old_prob + 1e-10)
    p1 = ratio * adv
    p2 = tf.clip_by_value(ratio, 1 - clip, 1 + clip) * adv
    
    loss = -tf.math.reduce_mean(tf.math.minimum(p1, p2) + entropy * -(prob * tf.math.log(prob + 1e-10)))
    return loss

class PPO:
    def __init__(self, env, epsilon=.99995, gamma=0.99, lam=0.95):
        self.env  = env
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.epsilon = epsilon
        self.epsilon_min = 0.001
        self.gamma = gamma
        self.lam = lam
        self.memory = deque(maxlen=500)

        self.actor, self.critic = self.create_model()

    def create_model(self):
        input_layer = Input(shape=self.state_shape)
        dense1 = Dense(64, activation='relu')(input_layer)
        dense2 = Dense(64, activation='relu')(dense1)
        probs = Dense(self.action_size, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        actor = Model(inputs= input_layer, outputs=probs)
        actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00035), loss=ppo_loss)

        critic = Model(inputs= input_layer, outputs=values)
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")

        return actor, critic

    def v(self, state):
        return self.critic.predict(state.reshape(1, *state.shape))[0]

    def get_action(self, state):
        state = state.reshape(1, *state.shape)
        policy = self.actor.predict(state).flatten()
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action, policy

    def remember(self, state, action, probs, reward, new_state, done):
        self.memory.append((state, action, probs, reward, new_state, done))

    def train(self, batch_size=32):
        if batch_size > len(self.memory):
            return
        minibatch = self.memory
        if len(minibatch) < 2:
            return

        states = []
        actions = []
        probs = []
        y = []
        adv = []
        gae = 0
        for (s, a, p, r, s2, done) in reversed(minibatch):
            v = self.v(s)
            if done:
                target = -10
                delta = target - v
            else:
                v2 = self.v(s2)[0]
                target = r + self.gamma * v2
                delta = r + self.gamma * v2 - v
            
            gae = delta + self.gamma * self.lam * gae
            
            states.append(s)
            actions.append(a)
            probs.append(p)
            y.append(target)
            adv.append(gae)
        
        states = np.vstack(states)
        self.critic.fit(states, y, epochs=8, verbose=0)

        adv = (adv - np.mean(adv)) / np.maximum(np.std(adv), 1e-10)
        probs = np.vstack(probs)
        y_actual = np.empty(shape=(len(states), 2 + self.action_size))
        y_actual[:, 0] = actions
        y_actual[:, 1] = adv.flatten()
        for i in range(self.action_size):
            y_actual[:, 2 + i] = probs[:, i]
        loss = self.actor.fit(states, y_actual, epochs=8, verbose=0)

        self.memory = deque(maxlen=500)

    def act(self, state):
        self.epsilon *= self.epsilon
        self.epsilon = max(self.epsilon, self.epsilon_min)
        action, probs = self.get_action(state)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample(), probs
        return action, probs