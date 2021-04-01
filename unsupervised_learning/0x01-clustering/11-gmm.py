#!/usr/bin/env python3
""" calcs gaussian mixture model from dataset """
import sklearn.mixture


def gmm(X, k):
    """ calcs GMM from dataset
        X: np arr (n, d) dataset
        k: num clusters
        Returns: pi, m, S, clss, bic (all np arrays)
          pi: (K,) cluster priors)
          m: (k, d) centroid means
          S: (k, d, d) covariance matrices
          clss: (n,) cluster indices
          bic: (kmax - kmin + 1,) BIC value for eahc cluster size tested
    """
    gm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gm.weights_
    m = gm.means_
    S = gm.covariances_
    clss = gm.predict(X)
    bic = gm.bic(X)
    return pi, m, S, clss, bic
