#!/usr/bin/env python3
""" determines next action with epsilon greedy """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ usues spsilon greedy to determine next action
        Q: the q table
        state: current state
        epsilon: epsilon to use for calculaton
        Returns next action index
    """
    p = np.random.uniform()
    if p > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(Q.shape[1])
