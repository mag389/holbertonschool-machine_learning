#!/usr/bin/env python3
""" bayes optimization """
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """ calculates next best sample location
            Uses Expected improvement acquisition function
            Returns: X_next, EI
              X_next: np arr (1,) of next best sample point
              EI: np arr (ac_samples,) of expected improvement of each
                potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        s = len(mu)
        mu_sample_opt = np.max(self.gp.Y)
        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
            mu_sample_opt = 2 * mu - mu_sample_opt
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xsi
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
        """ i was on the right track but didn't end up using this code
            instead determined min vs max at different point
        if self.minimize == False:
            mu_max = np.max(self.gp.Y)
        else:
            mu_min = np.min(self.gp.Y)
        """

    def optimize(self, iterations=100):
        """ optimizes the black box funciton
            iterations: max number of iterations to perform
            if next proposed point has already been sample stop early
            Returns: X-opt, Y_opt
              X_opt: np arr (1,) the optimal point
              Y_opt: np arr (1,) the optimal function value
        """
        visited = []
        for i in range(iterations):
            X_next, ei = self.acquisition()
            if X_next in visited:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            visited.append(X_next)
        if self.minimize is True:
            opt = np.argmin(self.gp.Y)
        else:
            opt = np.argmax(self.gp.Y)
        return self.gp.X[opt], self.gp.Y[opt]
