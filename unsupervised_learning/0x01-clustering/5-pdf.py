#!/usr/bin/env python3
""" calculates pdf of gaussian dist """
import numpy as np


def pdf(X, m, S):
    """ calulates probability density of given gaussian dist
        X: np array (n, d) of data points to evaluate pdf of
        m: npy array (d,) of mean of distribution
        S: np array (d, d) of covariance of distribution
        no loops or diag stuff
        Returns: P or None
          P: np array (n,) pdf values for each data point
        values in P are min 1e-300
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    n, d = X.shape
    if type(m) is not np.ndarray or len(m.shape) != 1 or m.shape[0] != d:
        print("first check")
        return None
    # if m.shape[0] != X.shape[1]:
    #     return None
    if type(S) is not np.ndarray or len(S.shape) != 2 or S.shape[0] != d:
        print("second check")
        return None
    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
    except Exception:
        return None
    if det == 0:
        return None

    norm_const = np.sqrt(((2 * np.pi) ** d) * det)
    """
    x_mu = X.T - m[:, np.newaxis]
    solve = np.linalg.solve(S, x_mu).T.dot(x_mu)
    return 1 / norm_const * np.exp(-(solve / 2))
    """
    x_mu = X.T - m[:, np.newaxis]

    part2 = np.multiply(np.matmul(inv, x_mu), x_mu)

    part2 = (-0.5) * np.sum(part2, axis=0)
    result = np.exp(part2) / norm_const

    return np.where(result < 1e-300, 1e-300, result)