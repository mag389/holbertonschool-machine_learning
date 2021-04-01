#!/usr/bin/env python3
""" finds best number of clusters for GMM
    uses bayesian informaion criterion
"""
import numpy as np
expectation_maximization = __import__('7-EM').expectation_maximization


def BIC(X, kmin, kmax, iterations=1000, tol=1e-5, verbose=False):
    """ finds ideal clusters
        X: np arr (n, d) data set
          n: number of data pts, d: dim's per data pt
        kmin: positive int of min number of clusters to check
        kmax: positive int of max number of clusters to check
        iterations: positive int containing max number of iterations for EM
        tol: non-neg float of em tolerance
        verbose: bool determing when to print
        1 loop
        Returns: best_k, bets_result, l, b or None, None, None, None
          best_k: best value for k based on it's bic
          best_result: tuple of (pi, m, S) output of ME
          l: np arr (kmax - kmin + 1) of log likelihood for each cluster
          b: np arr (kmax - kmin + 1) of bic value for each luster
            BIC = p * ln(n) - 2 * l
              p: number of params required for model
              n: number of data pts
              l: log likelihood of model
    """
    fail = (None, None, None, None)
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return fail
    if type(kmin) is not int or kmin < 1:
        return fail
    if type(kmax) is not int or kmax < 1 or kmax < kmin:
        return fail
    if type(iterations) is not int or iterations < 1:
        return fail
    if type(tol) is not float or tol < 0:
        return fail
    if type(verbose) is not bool:
        return fail
    ite = iterations
    n, d = X.shape
    ems = []
    logs = np.ones(kmax - kmin + 1)
    bics = np.ones(kmax - kmin + 1)
    # logs = np.array([])
    # bics = np.array([])

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, ite, tol, verbose)
        ems.append((pi, m, S, g, ll))
        # number of parameters: k*d means, k - 1 priors,
        # covariance mat terms: k * d * (d + 1) / 2
        p = k * d + (k - 1) + k * d * (d + 1) / 2
        logs[k - kmin] = ll
        bics[k - kmin] = (p * np.log(n) - 2 * ll)

    best_place = np.argmin(bics)
    best_k = np.arange(kmin, kmax + 1)[best_place]
    best_result = ems[best_place][0:3]
    return best_k, best_result, logs, bics
