#!/usr/bin/env python3
""" performing k-means """
import numpy as np


def kmeans(X, k, iterations=1000):
    """ performs k-means on a dataset
        X: the dataset: nd.ndarray (n, d)
          n: number of data points
          d: number of dimensions
        k: positive int of number of clusters
        iterations: positive int of max number of iterations to perform
        returns if no change in centroids
        initialize with multivariate uniform
        if cluster contains no data reinitialize it's centroid
        use numpy.random.uniform twice and at most 2 loops
        Return: C, clss or None, None on failure
          C: numpy.ndarray (k, d) of centroids
          clss: np.ndarray (n,) of ind of cluster in C each data point goes to
    """
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if not isinstance(X, np.ndarray):
        return None, None
    if len(X.shape) != 2:
        return None, None
    n, d = X.shape

    # init centroids and clss
    rmax = np.amax(X, axis=0)
    rmin = np.amin(X, axis=0)
    C = np.random.uniform(rmin, rmax, (k, d))
    clss = np.zeros(n)
    change = 0

    # perform clustering iterations
    for i in range(iterations):
        C_cpy = np.copy(C)

        # calculate classes
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        # distances = np.linalg.norm(X - C_cpy)
        clss = np.argmin(distances, axis=0)
        for j in range(k):
            # check no class is empty
            if len(X[clss == j]) == 0:
                C[j] = np.random.uniform(rmin, rmax, (1, d))
            else:
                # recalc centroid
                C[j] = np.mean(X[clss == j], axis=0)
        if np.array_equal(C_cpy, C):
            break
    return (C, clss)
