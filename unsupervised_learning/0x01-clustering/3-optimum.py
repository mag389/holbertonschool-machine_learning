#!/usr/bin/env python3
""" test for optimal number of clusters """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for optimum number of lcusters by varaince
        X: data set. np.ndarray (n, d)
          n: number of data poitns
          d: dimensions per data point
        kmin: positive int containing min number of clusters to check
        kmax: positive int containing max number of clusters to check
        iterations: positive integer containing max number of iters for kmeans
        should analyze at least 2 cluster sizes
        can use imports above
        at most 2 loops
        Returns: results, d_vars or None, None on failure
          results: list of outputs of K-means for each cluster size
          d_vars: list of difference in var from smallest cluster size for each
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is None:
        kmax = kmin + 3
    if not isinstance(kmax, int) or kmax < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    n, d = X.shape
    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        results.append(kmeans(X, k, iterations))
        vari = variance(X, results[-1][0])
        if k == kmin:
            small_var = vari
        d_vars.append(np.abs(vari - small_var))
    return results, d_vars
