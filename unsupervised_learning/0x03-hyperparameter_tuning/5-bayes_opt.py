#!/usr/bin/env python3
""" task 4 """


import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """ classe de performance de bayesian  """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        constructeur parametré
        :param f:
        :param X_init:
        :param Y_init:
        :param bounds:
        :param ac_samples:
        :param l:
        :param sigma_f:
        :param xsi:
        :param minimize:
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples, retstep=False)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        description
        """
        ml_sp, sg_spl = self.gp.predict(self.X_s)
        if self.minimize is True:
            value = np.min(self.gp.Y)
            retour = value - ml_sp - self.xsi
        else:
            value = np.max(self.gp.Y)
            retour = ml_sp - value - self.xsi
        with np.errstate(divide='warn'):
            Z = retour / sg_spl
            EI = retour * norm.cdf(Z) + sg_spl * norm.pdf(Z)

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        black  box
        :param iterations:
        """

        tab = []

        for i in range(iterations):
            suiv = self.acquisition()[0]
            if suiv in tab:
                break
            suiv_Y = self.f(suiv)
            self.gp.update(suiv, suiv_Y)
            tab.append(suiv)

        if self.minimize is False:
            idx = np.argmax(self.gp.Y)
        else:
            idx = np.argmin(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
