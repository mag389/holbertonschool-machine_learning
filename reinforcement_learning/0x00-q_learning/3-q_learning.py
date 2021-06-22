#!/usr/bin/env python3
""" performs q learning training """
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.5):
    """ performs Q-learning
        env: FrozenLakeEnv instance
        Q: np arr of Q table
        episodes: total episodes to train over
        max_steps: maxx steps per episode
        alpha: learning rate
        gamma: iscount rate
        epsilon: inital threshold for greedy epsilon
        min_epsilon: min value epsilon should go to
        epsilon_decay: decay rate for updating epsilon
          when agent falls in hole reward updates to -1
        Returns: Q, total_rewards
          Q: from above
          total_rewards: list containing rewards per episode
    """
    total_rewards = []
    for i in range(episodes):
        state = env.reset()
        done = False
        rewards_current_episode = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done and reward == 0:
                reward = -1
                # rewards_current_episode = -1
            # alternate q update
            """
            Q[state, action] = Q[state, action] * (1 - alpha) + \
                alpha * (reward + gamma * np.max(Q[new_state, :]))
            """
            # this is the one from live code session
            Q[state, action] += (alpha * (reward + gamma
                                            * np.max(Q[new_state])
                                            - Q[state, action]))
            # """
            state = new_state
            rewards_current_episode += reward
            if done and reward == 0:
                reward = -1
                # rewards_current_episode = -1
            if done:
                break
        # """
        epsilon = min_epsilon + \
            (epsilon - min_epsilon) * np.exp(-epsilon_decay * i)
        """
        epsilon = max(epsilon * (1-epsilon_decay), min_epsilon)
        """
        total_rewards.append(rewards_current_episode)
    return Q, total_rewards
