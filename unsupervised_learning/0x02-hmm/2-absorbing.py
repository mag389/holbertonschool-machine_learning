#!/usr/bin/env python3
""" determines absorbing chains """
import numpy as np


def absorbing(P):
    """ determines if markov chain is absorbing
        P: np arr (n, n) transition matrix
          P[i, j] is prob to go from i to j
          n: number of states
        Returns: True of absorbing or False on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    retmat = np.zeros(n)
    a_states = []
    for i in range(n):
        if P[i, i] == 1:
            retmat[i] == 1
            a_states.append(i)

    for i in range(n):
        if i in a_states:
            pass
        row = P[i]
        for col in range(n):
            if row[col] > 0 and col in a_states:
                retmat[i] = 1

    last = retmat.copy()
    while True:
        for i in range(n):
            if retmat[i] == 1:
                pass
            row = P[i]
            for col in range(n):
                if row[col] > 0 and retmat[col] == 1:
                    retmat[i] = 1
        if (retmat == last).all():
            break
        if (retmat == 1).all():
            return True
    return (retmat == 1).all()
