#!/usr/bin/env python3
""" plays the game post training """
import numpy as np


def play(env, Q, max_steps=100):
    """ plays the game
        env: FrozenLakeEnv instance
        Q: np arr of Q table
        max_steps: max steps per episode
        displays each step pf the board
        always exploit Q-table
        Returns: reward
    """
    state = env.reset()
    done = False
    reward = 0
    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state])
        state, reward, done, info = env.step(action)
        # env.render()
        if done:
            break
    env.render()
    return reward
