#!/usr/bin/env python3
""" intra cluster variance """
import numpy as np


def variance(X, C):
    """ calculate intra-cluster variance for a data set
        X: np.ndarray (n, d) of data set
        C: np.ndarray (k, d) of centroid means for each cluster
          n: number of data points
          d: dimensions per data point
          k: num clusters
        no loops
        Returns total variance (var) or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if C.shape[1] != X.shape[1]:
        return None
    n, d = X.shape
    k, _ = C.shape
    distances = np.sqrt(np.sum((X - C[:, np.newaxis])**2, axis=-1))
    varis = np.amin(distances, axis=0)
    return np.sum(varis**2)
