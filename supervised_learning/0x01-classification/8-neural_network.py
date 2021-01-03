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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
