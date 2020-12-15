#!/usr/bin/env python3
""" the expenential class file """


class Exponential:
    """ represents an exponential distribution of random variables"""
    def __init__(self, data=None, lambtha=1):
        """ basic constructor from past data"""
        self.lambtha = float(lambtha)
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            inverlamb = sum(data) / len(data)
            self.lambtha = 1 / inverlamb

    def pdf(self, x):
        """ calculates value of pdf for given time"""
        if x <= 0:
            return 0
        e = 2.7182818285
        return self.lambtha * e ** (-1 * self.lambtha * x)

    def cdf(self, x):
        """ calculates value for cdf of time period x """
        if x <= 0:
            return 0
        e = 2.7182818285
        return 1 - (e ** (-1 * self.lambtha * x))
