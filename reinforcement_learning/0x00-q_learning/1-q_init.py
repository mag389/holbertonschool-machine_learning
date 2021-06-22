#!/usr/bin/env python3
""" initialize q table """
import numpy as np


def q_init(env):
    """ initializes Q-table
        env: FrozenLakeEnv instance
        returns: Q-table as np arr of zeros
    """
    actions = env.action_space.n
    observs = env.observation_space.n
    return np.zeros((observs, actions))
