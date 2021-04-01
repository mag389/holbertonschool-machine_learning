#!/usr/bin/env python3
""" expectation step in EM algo for a GMM """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ calculates expectation step
        X: np array (n, d)
        pi: np array (k,) priors for each cluster
        m: np array (k, d) centroid means
        S: np array (k, d, d) cov matrix for each cluster
          n: number of data pts
          d: dimensions per pt
          k: number of clusters
        one loop
        Returns: g, l, or None, None
          g: np array (k, n) of posterior probs for each data pt per cluster
          l: total log likelihood
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None
    n, d = X.shape
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    g_old = np.ones((k, n))
    for i in range(k):
        # bayes rule to calculate posterior
        g_old[i] = pi[i] * pdf(X, m[i], S[i])
        # each g[i] must also eb divided by total probability of
        # that point irrespective of cluster (sum pi * pdf)
    g = g_old / np.sum(g_old, axis=0)

    # calculate log likelihood
    llh = np.sum(np.log(np.sum(g_old, axis=0)))

    return g, llh
