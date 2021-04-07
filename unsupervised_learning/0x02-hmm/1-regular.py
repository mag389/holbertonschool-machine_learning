#!/usr/bin/env python3
""" determins steady stats pros of markov chain """
import numpy as np


def regular(P):
    """ calculate steady state probabilities
        P: square 2d np arr (n, n) of transition mat
          P[i, j]: prob of going from i to j
          n: number of states
        Returns: np arr (1, n) containing steady state probs, or None
        uses two methods either eigs or QTQ
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.linalg.det(P) == 0:
        return None
    if not (P > 0).all():
        return None
    n = P.shape[0]
    q = (P - np.eye(n))
    ones = np.ones(n)
    q = np.c_[q, ones]
    QTQ = np.dot(q, q.T)
    bqt = np.ones(n)
    return np.linalg.solve(QTQ, bqt)
    return np.expand_dims(np.linalg.solve(QTQ, bqt), axis=0)

    a, b = np.linalg.eig(P.T)
    li = []
    for i in range(len(a)):
        if np.allclose(a[i], 1):
            li.append(i)
    if len(l) == 1:
        return np.abs(b[:, l[0]].T)/np.sum(np.abs(b[:, l[0]].T))
    else:
        return None
