#!/usr/bin/env python3
""" GaussianProcess file for representing gaussian process """
import numpy as np


class GaussianProcess():
    """ noiseless 1D gaussian process """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ initialize GaussianProcess object
            X_init: np arr (t, 1) of inputs sampled with black-box f'n
            Y_init: np arr (t, 1) of outputs of blackbox f'n for each input
              t: number of initial samples
            l: length parameter for kernel
            sigma_f: standard dev given to output of black-box f'n
            calculates K representing covariance matrix
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.Y)

    def kernel(self, X1, X2):
        """ calculates covariance kernel matrix between two matrices
            X1, X2: np arrays (m, 1) and (n, 1) respectively
            uses radial basis fucntion (RBF)
            Returns: covariance kernel matrix np arr (m, n)
        """
        sigma_f = self.sigma_f
        l = self.l
        m = X1.shape[0]
        n = X2.shape[0]
        sqdist = np.sum(X1**2, 1).reshape(-1, 1)\
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
