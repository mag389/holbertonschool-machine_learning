#!/usr/bin/env python3
""" training depp q net for atari breakout """
import gym
import numpy as np
import tensorflow.keras as k
import tensorflow.keras.layers as layers

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy, LinearAnnealedPolicy
from rl.callbacks import FileLogger


env = gym.make("Breakout-v0")
num_actions = env.action_space.n
state_size = env.observation_space.shape


def build_model(state_size, num_actions):
    """ build the keras model for deep learning """
    # inputs = layers.Input(shape=(84, 84, 4,))
    inputs = layers.Input(shape=(4,) + state_size)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)
    return k.Model(inputs=inputs, outputs=action)


model = build_model(state_size, num_actions)
model.summary()
"""
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=0.1, nb_steps=1000000)
"""
memory = SequentialMemory(limit=1000000, window_length=4)
agent = DQNAgent(model=model, policy=GreedyQPolicy(), nb_actions=num_actions,
                 memory=memory, nb_steps_warmup=50000)
agent.compile(k.optimizers.Adam(learning_rate=.00025), metrics=['mae'])
"""
agent.fit(env, nb_steps=10000, log_interval=1000, visualize=False, verbose=2)
"""
agent.load_weights('policy.h5')
agent.test(env, nb_episodes=10, visualize=False)
