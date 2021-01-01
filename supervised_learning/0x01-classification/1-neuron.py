#!/usr/bin/env python3
""" the first neuron class file"""


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
