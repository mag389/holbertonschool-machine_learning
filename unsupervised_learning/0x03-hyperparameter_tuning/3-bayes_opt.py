#!/usr/bin/env python3
""" bayes optimization """
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """ 1d bayesaian gaussian process """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """ initializer
            f: black box function to be optimizd
            X_init: np arr (t, 1) of inputs already sample with blackblox f'n
            Y_init: np arr (t, 1) of outputs of BB f'n
              t: number of initial samples
            bounds: tuple of (min, max) of bounds of the space to look for
              optimal point
            ac_samples: number of samples to analyze during acquisition
            l: length parameter for kernel
            sigma_f: standard deviation given to output of black box f'n
            xsi: oxploration-exploitation factor for acquisition
            minimize: bool to perform minimization (true) or optimization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        ac = ac_samples
        self.X_s = np.linspace(bounds[0], bounds[1], ac).reshape((ac, 1))
        self.xsi = xsi
        self.minimize = minimize
