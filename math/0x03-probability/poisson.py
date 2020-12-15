#!/usr/bin/env python3
"""the poisson class file"""


class Poisson:
    """ the poisson class for poisson random variables"""
    def __init__(self, data=None, lambtha=1.):
        """constructor for poisson class"""
        self.data = data
        self.lambtha = float(lambtha)
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            datasum = 0
            for point in data:
                datasum += point
            self.lambtha = datasum / len(data)

    def pmf(self, k):
        """ calculate pmf for given value k """
        e = 2.7182818285
        kfact = 1
        if k < 0:
            return 0
        k = int(k)
        for i in range(1, k + 1):
            kfact *= i
        return (self.lambtha ** k) * (e ** (-1 * self.lambtha)) / kfact

    def cdf(self, k):
        """ calculated cdf for value k """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        # print(self.pmf(k))
        e = 2.7182818285
        const = (e ** (-1 * self.lambtha))
        return self.pmf(k) + self.cdf(k - 1)
