#!/usr/bin/env python3
""" performs kmeans with sklearn """
import sklearn.cluster


def kmeans(X, k):
    """ performs kmeans using sklearn
        X: np arr (n, d)
          n: number of data pts, d: dims per pt
        k: num clusters
        Returns: C, clss
          C: np arr (k, d) means, clss: np arr (n,) index of clusters for C
    """
    means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return means.cluster_centers_, means.labels_
