# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:59:42 2024

@author: weichunwen
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# prompt: 測試程式改成若reward>=25就顯示25即可

import highway_env
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from gymnasium.envs.registration import register, registry
from tensorflow.keras import layers
import multiprocessing

# Register the Roundabout environment
def register_roundabout_env():
    if 'roundabout-v0' not in registry:
        register(
            id='roundabout-v0',
            entry_point='highway_env.envs:RoundaboutEnv',
        )

register_roundabout_env()

env_name = "roundabout-v0"
env = gym.make(env_name, render_mode='rgb_array')

class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.policy_logits = layers.Dense(num_actions)
        self.values = layers.Dense(1)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.policy_logits(x), self.values(x)

def preprocess(frame):
    if isinstance(frame, tuple):
        frame = frame[0]
    frame = frame.astype(np.float32) / 255.0
    if frame.ndim == 2:  # If the frame is already 2D
        return np.expand_dims(frame, axis=2)
    else:  # If the frame is 3D
        frame = frame[:, :, 0]  # Assuming we need the first channel only
        return np.expand_dims(frame, axis=2)

class A3C:
    def __init__(self, env_name, num_actions, lr=7e-4, num_episodes=1):
        self.env_name = env_name
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.global_model = ActorCritic(num_actions)
        self.global_model.build(input_shape=(None, 5, 5, 1))  # Assuming 5x5 state size
        self.rewards_queue = multiprocessing.Queue()

    def test(self, num_episodes=1, render=True):
        env = gym.make(self.env_name, render_mode='rgb_array')
        self.global_model.load_weights('a3c_250_weights.h5')

        for episode in range(num_episodes):
            state = preprocess(env.reset())
            done = False
            total_reward = 0
            while not done:
                if render:
                    env.render()
                logits, _ = self.global_model(state[np.newaxis, :])
                action = np.argmax(logits.numpy()[0])
                next_state, reward, done, truncated, info = env.step(action)
                state = preprocess(next_state)
                total_reward += reward
                if total_reward >= 25:
                    total_reward = 25
                    break
            print(f"Test Episode {episode} finished with Total Reward: {total_reward}")
        env.close()

a3c = A3C(env_name, env.action_space.n)
a3c.test()

