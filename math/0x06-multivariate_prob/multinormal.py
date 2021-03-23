#!/usr/bin/env python3
""" a class to represent multi normal """
import numpy as np


class MultiNormal:
    """ represents multinormal random variable """
    def __init__(self, data):
        """ creates multinormal instance """
        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape[0], data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.matmul(data - self.mean, data.T - self.mean.T)

    def pdf(self, x):
        """ calculates the pdf at a data point
            x: np.ndarray (d, 1) containing a data point
              d: number of dimensions
        """
        if type(x) != np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        d0, d1 = x.shape
        if d0 != d or d1 != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        adj_mean = x - self.mean
        pdf = 1.0 / (np.sqrt((2 * np.pi)**d * det))
        pdf1 = np.matmul((np.matmul(adj_mean.T, inv), adj_mean))
        pdf = pdf * np.exp(-0.5 * pdf1)
        return pdf[0][0]
