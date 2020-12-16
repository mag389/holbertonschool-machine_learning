#!/usr/bin/env python3
""" binomial distribution file"""


class Binomial:
    """ the binomial distribution class """

    def __init__(self, data=None, n=1., p=0.5):
        """constructor for binomial distribution
           n trials, p prob for success
        """
        if data is None:
            self.n = int(n)
            self.p = float(p)
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = 0
            for number in data:
                variance += (number - mean) ** 2
            variance /= (len(data))
            stddev = variance ** .5
            q = variance / mean
            p = 1 - q
            n = round(mean / p)
            self.p = mean / n
            self.n = n

    def pdf(self, x):
        """calculates pdf for x value"""
        pi = 3.1415926536
        e = 2.7182818285
        coeff = 1 / (self.stddev * (2 * pi) ** .5)
        expo = -.5 * ((x - self.mean) / self.stddev) ** 2
        pdfval = coeff * e ** (expo)
        return pdfval

    @staticmethod
    def erf(x):
        """return the erf for x """
        pi = 3.1415926536
        coeff = 2 / (pi ** .5)
        series = x - x ** 3 / 3 + x ** 5 / 10 - x ** 7 / 42 + x ** 9 / 216
        return float(coeff * series)

    def cdf(self, x):
        """the cdf for the normal distribution object """
        pi = 3.1415926536
        e = 2.7182818285
        erfval = (x - self.mean) / (self.stddev * 2 ** .5)
        y = erfval
        series = (1 + (y - y ** 3 / 3 + y ** 5 / 10 - y ** 7 / 42
                       + y ** 9 / 216) * 2 / (pi ** .5)) / 2
        cdfval = (.5 + series)
        return series
