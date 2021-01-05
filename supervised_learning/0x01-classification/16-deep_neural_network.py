#!/usr/bin/env python3
""" the deep neural network class file for defining deep neural network """


import numpy as np


class DeepNeuralNetwork:
    """ the deep neural network class
        a neural network with a variable number of layers
    """

    def __init__(self, nx, layers):
        """ DeepNeuralNetwork constructor
            args:
                nx: int >=1 number of input features
                layers: list of numbers: number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) is 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(1, len(layers) + 1):
            if not isinstance(layers[i - 1], int) or layers[i - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
            """he at al."""
            li = layers[i - 2]
            if i is 1:
                li = nx
            self.weights["W" + str(i)] = np.random.randn(layers[i - 1],
                                                         li) * np.sqrt(2 / li)
            self.weights["b" + str(i)] = np.zeros((layers[i - 1], 1))
        # self.weights["W" + str(len(layers))] = np.random.randn(len(layers),
        #     layers[len(layers) - 1]) * np.sqrt(2 / layers[len(layers) - 1])
