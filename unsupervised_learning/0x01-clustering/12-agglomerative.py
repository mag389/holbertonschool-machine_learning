#!/usr/bin/env python3
""" agglomerative clustering on datasets """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ performs agglomerative clustering
        X: np arr (n, d) dataset
        dist: maximum cophenetic distance for all clusters
        uses ward linkage
        displays dendogram with each cluster in different color
        Returns: clss np arr (n,) or cluster indices
    """
    sch = scipy.cluster.hierarchy
    # sch.ward(y) takes condensed matrix
    # cophenet : calculates cophenetic distance
    # dendogram: plot hierarchical clustering as dendogrm
    Z = sch.ward(X)
    clss = sch.fcluster(Z, t=dist, criterion='distance')
    pl = sch.dendrogram(Z, color_threshold=dist)
    plt.show()
    return clss
