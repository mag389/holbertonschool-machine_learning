#!/usr/bin/env python3
""" normalize the matrix """
import numpy as np


def normalize(X, m, s):
    """ normalizes a matrix
        X: np.ndarray (d, nx) d data points, nx features
        m: np.ndarray (nx,) mean of all features of X
        s: np.ndarray (nx,) std dev of all features of X
        Returns: normalized matrix
    """
    X -= m
    X /= s
    return X
