#!/usr/bin/env python3
""" determins steady stats pros of markov chain """
import numpy as np


def regular(P):
    """ calculate steady state probabilities
        P: square 2d np arr (n, n) of transition mat
          P[i, j]: prob of going from i to j
          n: number of states
        Returns: np arr (1, n) containing steady state probs, or None
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    a, b = np.linalg.eig(P.T)
    l = []
    for i in range(len(a)):
        if np.allclose(a[i], 1):
            l.append(i)
    if len(l) == 1:
        return np.abs(b[:, l[0]].T)/np.sum(np.abs(b[:, l[0]].T))
    else:
        return None
