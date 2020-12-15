#!/usr/bin/env python3
""" normal distribution file"""


class Normal:
    """ the normal distribution class """

    def __init__(self, data=None, mean=0., stddev=1.):
        """constructor for normal distribution"""
        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
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
            self.mean = mean
            self.stddev = stddev

    def z_score(self, x):
        """ calculates z score from x values"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calculate x value from z"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """calculates pdf for x value"""
        pi = 3.1415926536
        e = 2.7182818285
        coeff = 1 / (self.stddev * (2 * pi) ** .5)
        expo = -.5 * ((x - self.mean) / self.stddev) ** 2
        pdfval = coeff * e ** (expo)
        return pdfval
