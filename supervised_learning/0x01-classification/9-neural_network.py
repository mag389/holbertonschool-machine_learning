#!/usr/bin/env python3
""" The neural network class file for defining a neural network
    this network has one hidden layer
"""


import numpy as np


class NeuralNetwork:
    """the NeuralNetwork class
       single hidden layer
    """
    def __init__(self, nx, nodes):
        """ NeuralNetwork constructor
            args:
                nx: int, >=1, number of input features
                nodes: int, >=1, number of hidden layer nodes
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ getter for hidden layer weights vector shape(nodes, nx)"""
        return self.__W1

    @property
    def b1(self):
        """ getter for hidden layer bias vector shape(1, nodes)"""
        return self.__b1

    @property
    def A1(self):
        """ getter for activated output for hidden layer """
        return self.__A1

    @property
    def W2(self):
        """getter for weights vector of output neuron shape(1, nodes)"""
        return self.__W2

    @property
    def b2(self):
        """getter for bias for output neuron"""
        return self.__b2

    @property
    def A2(self):
        """getter for activated output for output neuron (prediction)"""
        return self.__A2
