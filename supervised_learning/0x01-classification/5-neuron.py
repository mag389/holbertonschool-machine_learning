#!/usr/bin/env python3
""" the neuron class file
    defines a single neuron for binary classification
"""


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
        """calculates forward propagation for a neuron
        X is a numpy.ndarray with shape nc by m that contains input data
            nx is number of input features to the neuron
            m is the number of examples
        update private attribute __A using sigmoid activation function
        return private instance attribute __A
        """
        Y = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp((-1) * Y))
        return self.A

    def cost(self, Y, A):
        """calculates cost of the model with logistic regression
           Y is a 1 by m np.ndarray with labels for input data
           A is a 1 by m np.ndarray containing activated output
               of neurons for each example
           returns the cost
           J is cost function, or average of loss(error) for each data sample
           J results in (1,1) matrix
        """
        m = len(Y[0])
        J = -1 / m * (np.matmul(Y, np.log(A).T) +
                      np.matmul((1-Y), np.log(1.0000001 - A).T))
        return J[0][0]

    def evaluate(self, X, Y):
        """ evaluates a neuron's predictions
            X is a (nx, m) np.ndarray with input data
                nx is number of input features
                m is number of examples
            Y is (1, m) np.ndarray with correct labels
            returns neuron's prediction and cost of the network
                predicion is (1. m) np.ndarray with predicted labels
                labels 1 if output >= 0.5 else 0
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.A)
        return np.round(self.A).astype(np.int), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron
            X is np.ndarray (nx, m)
                nx is number of input features
                m is number of examples
            Y is np.ndarray (1, m) with the correct labels
            A is np.ndarray (1, m) with activated output of neuron for each ex
            alpha is learning rate
            updates private attributes __W and __b
                __W is the weights
                __b is the bias
            dw and db are the gradients of the cost function with respect
                to W and b respectiuvely
        """
        m = len(X[0])
        dw = np.matmul(X, (A - Y).T) / m
        db = np.sum((A - Y)) / m
        self.__W = self.__W - alpha * (dw.T)
        self.__b = self.b - alpha * (db)
