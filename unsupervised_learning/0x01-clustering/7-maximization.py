#!/usr/bin/env python3
""" maximization step of EM of GMM """
import numpy as np


def maximization(X, g):
    """ calucaltes maximization step of expectation-maximization algo
        X: np array (n, d) data set
        g: np array (k, n) of posterior probabilities
          n: number of data pts
          d: dim's per data pt
          k: num clusters
        1 loop max
        Returns: pi, m, S or None, None, None on failure
          pi: np arr (k,) of updated priors
          m: np arr (k, d) of updated centroid means
          S: np arr (k, d, d) of updated cov matrices per cluster
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if n != g.shape[1]:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), np.ones(n)).all():
        return None, None, None

    pi = np.ones((k,))
    m = np.ones((k, d))
    S = np.ones((k, d, d))

    for i in range(k):
        # first update each prior
        pi[i] = np.sum(g[i]) / n

        # update means (g * X accounting for dimensions etc)
        m[i] = np.sum(g[i, :, np.newaxis] * X, axis=0) / np.sum(g[i])

        # update cov matrix
        num = np.dot(g[i] * (X - m[i]).T, (X - m[i]))
        den = np.sum(g[i])
        S[i] = num / den
    return pi, m, S
