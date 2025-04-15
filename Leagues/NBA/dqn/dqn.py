import random
from collections import deque

import numpy as np
import tensorflow as tf


class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(action_size, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)


class Agent:
    def __init__(self, state_size,
                 action_size,
                 learning_rate=0.0001,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 decay=0.995,
                 batch_size=65,
                 memory_size=100000):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = decay
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.set_weights(self.q_network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def select_afction(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)[None, :]
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.numpy())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done = zip(*batch)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        # Compute target Q values
        next_q_values = self.target_network(next_states)
        target_q_values = rewards + self.gamma * \
            tf.reduce_max(next_q_values, axis=1) * (1-done)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values = tf.reduce_sum(q_values * tf.one_hot(tf.cast(actions, tf.int32), self.action_size), axis=1)

            loss = tf.keras.losses.MSE(target_q_values, q_values)

        #optimize
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        #decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())