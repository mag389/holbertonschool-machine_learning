#!/usr/bin/env python3
""" the third neuron class file"""


import numpy as np


class Neuron:
    """the neuron class"""
    def __init__(self, nx):
        """ the constructor, nx is number of input features"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ getter for neuron weights vector """
        return self.__W

    @property
    def b(self):
        """ getter for bias for the neuron starts as 0"""
        return self.__b

    @property
    def A(self):
        """ getter for activated output of neuron. initialized as 0"""
        return self.__A

    def forward_prop(self, X):
        """defines a single neuron performing binary classification
        X is a numpy.ndarray with shape nc by m that contains input data
            nx is number of input features to the neuron
            m is the number of examples
        update private attribute __A using sigmoid activation function
        return private instance attribute __A
        """
        Y = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp((-1) * Y))
        return self.A
