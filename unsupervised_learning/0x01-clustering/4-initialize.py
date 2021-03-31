#!/usr/bin/env python3
""" initialize Gaussian Mixture Model """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ inits GMM
        X: np array (n, d) of data set
          n: number of poitns, d: dims per point
        k: positive int: number of clusters
        no loops
        Returns: pi, m, S or None, None, None
          pi: np array (k,) priors for each cluster, evenly init
          m: np array (k, d) centroid means for each cluster
            initialized with k-means
          S: np array (k, d, d) covaraince matrix for each cluster
            initialized as identity matrix
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k < 1:
        return None, None, None
    n, d = X.shape
    pi = np.ones(k) / k
    m = kmeans(X, k)[0]
    S = np.array([np.identity(d)] * k)
    # S = np.repeat((np.identity(d)[:, np.newaxis]), k)
    return pi, m, S
