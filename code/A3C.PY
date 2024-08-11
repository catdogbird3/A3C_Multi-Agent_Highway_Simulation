import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:39:23 2024

@author: catdo
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
import numpy as np
import gymnasium as gym
import threading
import multiprocessing
from tensorflow.keras import layers
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)])
    except RuntimeError as e:
        print(e)
# Register the Roundabout environment
def register_roundabout_env():
    from gymnasium.envs.registration import register, registry
    if 'roundabout-v0' not in registry:
        register(
            id='roundabout-v0',
            entry_point='highway_env.envs:RoundaboutEnv',
        )

register_roundabout_env()

env_name = "roundabout-v0"
env = gym.make(env_name, render_mode='rgb_array')
print("Action Space:", env.action_space)
print("State Space:", env.observation_space)

#定義 ActorCritic 模型類別：這部分包含模型的神經網絡結構，用於計算策略(logits)和價值(values)
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
#定義 A3C 類別：該類別負責訓練過程，包括創建多個工作線程(worker)，每個線程會各自與環境進行交互，並將經驗積累到全局模型中。
class A3C:
    def __init__(self, env_name, num_actions, lr=7e-4, num_episodes=10):
        self.env_name = env_name
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.global_model = ActorCritic(num_actions)
        self.global_model.build(input_shape=(None, 5, 5, 1))  # Assuming 5x5 state size
        self.global_model.summary()
        self.rewards_queue = multiprocessing.Queue()

    def train(self):
        envs = [gym.make(self.env_name, render_mode='rgb_array') for _ in range(4)]  # 限制線程數量為 4
        workers = [Worker(env, self.global_model, self.optimizer, self.num_actions, self.num_episodes, self.rewards_queue) for env in envs]

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        all_rewards = []
        while not self.rewards_queue.empty():
            all_rewards.append(self.rewards_queue.get())

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(all_rewards)), all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.grid(True)
        plt.show()

        self.global_model.save_weights("a3c_weights.h5")
#定義 Worker 類別：每個 Worker 類別實例代表一個工作線程，負責與環境交互並更新本地模型和全局模型。
class Worker(threading.Thread):
    def __init__(self, env, global_model, optimizer, num_actions, num_episodes, rewards_queue, gamma=0.99, max_steps=500):
        super().__init__()
        self.env = env
        self.global_model = global_model
        self.optimizer = optimizer
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.local_model = ActorCritic(num_actions)
        self.local_model.build(input_shape=(None, 5, 5, 1))  # Assuming 5x5 state size
        self.rewards_queue = rewards_queue
        self.max_steps = max_steps

    def run(self):
        for episode in range(self.num_episodes):
            state = preprocess(self.env.reset())
            done = False
            total_reward = 0
            step = 0
            while not done and step < self.max_steps:
                logits, _ = self.local_model(state[np.newaxis, :])
                action = np.random.choice(self.num_actions, p=tf.nn.softmax(logits).numpy()[0])
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = preprocess(next_state)
                state = next_state
                total_reward += reward
                step += 1
                if step % 10 == 0:
                    print(f"Thread {self.name} Episode {episode} Step {step} Total Reward: {total_reward}")
            print(f"Thread {self.name} Episode {episode} finished with Total Reward: {total_reward}")
            self.rewards_queue.put(total_reward)

a3c = A3C(env_name, env.action_space.n, num_episodes=250)
a3c.train()

