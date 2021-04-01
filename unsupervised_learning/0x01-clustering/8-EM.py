#!/usr/bin/env python3
""" full expecttion maximizatino algo """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ performs exp-max algorithm
        X: np arr (n, d) data set
          n: number of data pts, d: dim's per data pt
        k: positive int, num of clusters
        iterations: positiv int, max number if iterations to perform
        tol: non-neg float: tolerance of log likelihood used to determine
            early stopping (if diff <= tol stop)
        verbose: if true print message every ten iterations and after last
            prints message about iterations num and log likelihood
        Returns: pi, m, S, g, l    or None, None, None, None, None on failure
          pi: np arr (k,) priors for each cluster
          m: np arr (k, d) centroid means for each cluster
          S: np arr (k, d, d) cov matrices for each cluster
          g: np arr (k, n) probs for each data pt in each cluster
          l: log likelihood of the model
    """
    fail = (None, None, None, None, None)
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return fail
    if type(k) is not int or k < 1:
        return fail
    if type(iterations) is not int or iterations < 1:
        return fail
    if type(tol) is not float or tol < 0:
        return fail
    if type(verbose) is not bool:
        return fail
    n, d = X.shape
    pi, m, S = initialize(X, k)
    l_old = 0
    for i in range(iterations):
        g, ll = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)
        if abs(ll - l_old) <= tol:
            break
        if verbose and i % 10 == 0:
            llr = round(ll, 5)
            print("Log Likelihood after {} iterations: {}".format(i, llr))
        l_old = ll
    if verbose:
        llr = round(ll, 5)
        print("Log Likelihood after {} iterations: {}".format(i, llr))
    return pi, m, S, g, ll
