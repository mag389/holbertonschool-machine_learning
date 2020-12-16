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

    def pmf(self, k):
        """calculates pmf for k successes"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        n = self.n
        p = self.p
        nfact = 1
        for i in range(k, n + 1):
            nfact *= i
        kfact = 1
        for i in range(1, n + 1):
            kfact *= i * p
        nkfact = 1
        for i in range(1, n - k + 1):
            nkfact *= i * (1 - p)
        # print(nfact, kfact, nkfact, sep="\n")
        # print("the combin is:", nfact / (kfact * nkfact))
        # print(p ** k, .6 ** 30)
        retval = 1
        for i in range(1, n + 1):
            retval *= i
            if i <= k:
                retval /= i
                retval *= self.p
            if i <= n - k:
                retval /= i
                retval *= (1 - self.p)
        return retval

    def cdf(self, k):
        """the cdf for the binomial distribution for k variable """
        retval = 0
        for i in range(0, k + 1):
            retval += self.pmf(i)
        return retval
