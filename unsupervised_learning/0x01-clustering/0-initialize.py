#!/usr/bin/env python3
""" initialize cluster centroids for k-means """
import numpy as np


def initialize(X, k):
    """ initialize cluster centroids
        X: np.ndarray (n, d) of dataset for clustering
          n: number of data points
          d: number of dimensions of eahc data point
        k: positive int containing number of clusters
        cluster centroids are initialized with multivariate uniform dist
          min values for dist should be minimul values of X along each dim
          max value for dist should be max value of X along each dimension
          uses numpy.random.uniform only once
        no loops
        Returns: np.ndarray (k, d) of initialized centroids for each cluster
          or None on failure
    """
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray):
        return None
    if len(X.shape) != 2:
        return None
    n, d = X.shape
    rmax = np.amax(X, axis=0)
    rmin = np.amin(X, axis=0)
    return np.random.uniform(rmin, rmax, (k, d))
