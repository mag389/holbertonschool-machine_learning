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

    def forward_prop(self, X):
        """ Calculates forward propagation of neural network
            X is np.ndarray (nx, m) of input data
                nx in number of input features
                m is number of examples
            uses sigmoid activation function
            returns self.__A1, self.__A2
        """
        Y1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp((-1) * Y1))
        Y2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp((-1) * Y2))
        return self.A1, self.A2

    def cost(self, Y, A):
        """ calculates cost of the model with logistic regression
            Y is (1, m) np.ndarray with correct labels for input data
            A is (1, m) np.ndarray with activated output of neuron
            return: the cost or avg loss(error) for each sample
        """
        m = len(Y[0])
        J = (-1 / m) * (np.matmul(Y, np.log(A).T) +
                        np.matmul((1-Y), np.log(1.0000001 - A).T))
        return J[0][0]

    def evaluate(self, X, Y):
        """ evaluates the network's predictions
            X is (nx, m) np.ndarray of input data
                nx is number of input features, m is number of examples
            Y is (1, m) np.ndarray of correct labels
            returns neurons prediction and cost
                prediction os np.ndarray (1, m) with predict labels
                label is 1 if output >=0.5 else 0
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.A2)
        return np.round(self.A2).astype(np.int), cost
