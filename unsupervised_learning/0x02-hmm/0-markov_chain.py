#!/usr/bin/env python3
""" probability of markov chain after specific number of iterations """
import numpy as np


def markov_chain(P, s, t=1):
    """ determines probabilities of markov chain after t iters
        P: square 2D np arr (n, n) representing transition martrix
          P[i, j] is prob of going from i to j
          n: number of states
        s: np arr (1, n) representing probability of starting in each state
        t: number of iterations
        Returns: np arr (1, n) of prob of being in a specific state
          None on failure
    """
    if type(P) is not np.ndarray or P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if type(s) is not np.ndarray or s.shape[1] != n:
        return None
    for i in range(t):
        s = s.dot(P)
    return s
