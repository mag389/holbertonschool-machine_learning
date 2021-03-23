#!/usr/bin/env python3
""" mean and cov of data set """
import numpy as np


def mean_cov(X):
    """ calcs mean and covariance of a data set
        X: np ndarray (n, d)
          n: number of points, d: number of dimensions of each data point
        Returns: mean, cov
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n = X.shape[0]
    d = X.shape[1]
    mean = np.mean(X, axis=0, keepdims=True)
    cov = (1 / (n - 1)) * np.matmul(X.T - mean.T, X - mean)
    return (mean, cov)
