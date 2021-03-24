#!/usr/bin/env python3
""" pca svd implementation """
import numpy as np


def pca(X, ndim):
    """ perform pca on X
        X: np.ndarray (n, d)
          n: number of data points
          d: number of dimensions in each point
          all dimension have mean of 0 across all data points
        ndim: new dimensionality of transformed X
        Returns: matrix T of transformation of X
          T: (d, ndim) nd is new dimensionality
        @ is matmul
        must do svd of X_m instead of svd of x
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    # traditional notation decomp(A) = U (sigma) VT = (u * s) @ vh
    W = vh[0:ndim].T
    # X_m = X - np.mean(X, axis=0)
    return np.matmul(X_m, W)
