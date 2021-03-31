#!/usr/bin/env python3
""" test for optimal number of clusters """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
        tests for optimum number of lcusters by varaince
        X: data set. np.ndarray (n, d)
          n: number of data poitns
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax < 1 or kmax < kmin + 1:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    # n, d = X.shape
    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        vari = variance(X, C)
        if k == kmin:
            small_var = vari
        d_vars.append(small_var - vari)
    return results, d_vars
