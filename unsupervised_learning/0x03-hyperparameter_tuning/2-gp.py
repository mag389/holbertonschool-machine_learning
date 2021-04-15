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
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ calculates covariance kernel matrix between two matrices
            X1, X2: np arrays (m, 1) and (n, 1) respectively
            uses radial basis fucntion (RBF)
            Returns: covariance kernel matrix np arr (m, n)
        """
        # kernel error
        """
        somme = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        retrait = somme + (-2 * np.dot(X1, X2.T))
        final = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * retrait)
        return final
        """
        # end
        sqdist = np.sum(X1**2, 1).reshape(-1, 1)\
            + np.sum(X2**2, 1) - 2 * np.matmul(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """ predicts the mean and std dev of pts in gaussian process
            X_s: np arr (s, 1) of all pts whose mean and std dev to calculate
              s: number of sample points
            Returns: mu, sigma
              mu: np arr (s,) of mean for each pt in X_s
              sigma: np arr (s,) of variance for each pt in X_s
        """
        # making sure this is correct
        Krnl_ss = self.kernel(X_s, X_s)
        Krnl_inv = np.linalg.inv(self.K)
        Krnl_s = self.kernel(self.X, X_s)
        # this line in mine was the error
        conv_s = Krnl_ss - Krnl_s.T.dot(Krnl_inv).dot(Krnl_s)
        sigma = np.diagonal(conv_s)

        multi = Krnl_s.T.dot(Krnl_inv).dot(self.Y)
        multi = np.reshape(multi, -1)

        return multi, sigma
        # end
        s = X_s.shape[0]
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)  # + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape((s,))
        sigma = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu, np.diag(sigma)

    def update(self, X_new, Y_new):
        """ updates the gaussian object instance
            X_new, Y_new: np arrays (1,) of new sample point and new sample f'n
            updates X, Y, K
        """
        # update error perhaps?
        self.X = np.concatenate((self.X, X_new[:, np.newaxis]), axis=0)
        self.Y = np.append(self.Y, Y_new)
        self.Y = np.reshape(self.Y, (-1, 1))
        self.K = self.kernel(self.X, self.X)
        return
        # end
        self.X = np.append(self.X, X_new.reshape((1, 1)), 0)
        self.Y = np.append(self.Y, Y_new.reshape((1, 1)), 0)
        self.K = self.kernel(self.X, self.X)
