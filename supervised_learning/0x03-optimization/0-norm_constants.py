#!/usr/bin/env python3
""" normalize the constants of a matrix """


import numpy as np
import tensorflow as tf


def normalization_constants(X):
    """ normalize constants in matrix
        X: np.ndarray (m, nx) m data points and nx number of features
        Returns: mean and standard deviation respectively
    """
    return X.mean(axis=0), X.std(axis=0)
