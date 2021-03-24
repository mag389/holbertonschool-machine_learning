#!/usr/bin/env python3
""" pca svd implementation """
import numpy as np


def pca(X, var=0.95):
    """ perform pca on X
        X: np.ndarray (n, d)
          n: number of data points
          d: number of dimensions in each point
          all dimension have mean of 0 across all data points
        var: fraction of variance pca should maintian
        Returns: Weight matrix W that maintains var frac of X's variance
          W: (d, nd) nd is new dimensionality
        @ is matmul
    """
    u, s, vh = np.linalg.svd(X)
    # traditional notation decomp(A) = U (sigma) VT = (u * s) @ vh
    tot_var = np.cumsum(s)
    req_var = tot_var[-1] * var
    for i in range(len(tot_var)):
        if tot_var[i] < req_var:
            continue
        break
    return vh[:i + 1].T
