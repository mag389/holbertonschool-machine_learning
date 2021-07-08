#!/usr/bin/env python3
""" performs monte carlo algo """
import gym
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """ performs monte carlo algorithm
        env: the openai environment instance
        V: np arr (s,) of value estimate
          s: state space
        policy: function that takes in a state and returns the enxt action
          to take
        episodes: total numer of episodes to train over
        max_steps: max steps per episode
        alpha: the learning rate
        gamma: the discount rate
        Returns: V, the updates value estimate
    """
    for episode in range(episodes):
        # for each episode reset state and run trial
        state = env.reset()
        done = False
        reward = 0
        rewards = [0]
        states = [state]
        for step in range(max_steps):
            # run trial up to possible max steps
            action = policy(state)
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            if done:
                break
        # with the updated results, perform MC
        G = 0
        visited_states = []
        for i in range(len(states) - 2, -1, -1):
            G = gamma * G + rewards[i + 1]
            state = states[i]
            # visited_states.append(state)
            # if state not in visited_states:
            if state not in states[0:i]:
                V[state] = V[state] + alpha * (G - V[state])
                visited_states.append(state)
    return V
