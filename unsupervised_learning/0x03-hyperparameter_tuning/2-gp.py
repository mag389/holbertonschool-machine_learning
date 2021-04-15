#!/usr/bin/env python3
""" task 0"""


import numpy as np


class GaussianProcess():
    """ classe description """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        constructeur
        :param X_init:
        :param Y_init:
        :param l:
        :param sigma_f:
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calcule de covariance
        :param X1:
        :param X2:
        """
        somme = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        retrait = somme + (-2 * np.dot(X1, X2.T))
        final = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * retrait)
        return final

    def predict(self, X_s):
        """
        prediction de la moyenne
        :param X_s:
        """
        Krnl_ss = self.kernel(X_s, X_s)
        Krnl_inv = np.linalg.inv(self.K)
        Krnl_s = self.kernel(self.X, X_s)

        conv_s = Krnl_ss - Krnl_s.T.dot(Krnl_inv).dot(Krnl_s)
        sigma = np.diagonal(conv_s)

        multi = Krnl_s.T.dot(Krnl_inv).dot(self.Y)
        multi = np.reshape(multi, -1)

        return multi, sigma

    def update(self, X_new, Y_new):
        """
        mise a jour
        :param X_new:
        :param Y_new:
        """
        self.X = np.concatenate((self.X, X_new[:, np.newaxis]), axis=0)
        self.Y = np.append(self.Y, Y_new)
        self.Y = np.reshape(self.Y, (-1, 1))
        self.K = self.kernel(self.X, self.X)
