#!/usr/bin/env python3
""" shuffles data in two matrices """
import numpy as np


def shuffle_data(X, Y):
    """ shuffles data in two matrices
        X, Y: np.ndarrays (m, nx) and (m, ny) to shuffle
            m data points
            nx, ny number of features respectively
        Returns: shuffled X and Y matrice
    """
    state = np.random.get_state()
    X = np.random.permutation(X)
    np.random.set_state(state)
    return X, np.random.permutation(Y)
