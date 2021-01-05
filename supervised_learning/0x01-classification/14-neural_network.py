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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ calculates one pass of gradient descent on the neural network
            X is (nx, m) np.ndarray with input data
            Y ia (1, m) np.ndarray with correct labels for input data
            A1 is output of hidden layer
            A2 is predicted output
            alpha is learning rate
                nx is number of input features per neuron
                m is number of examples
            updates: __W1, __b1, __W2, __b2
        """
        m = len(X[0])
        # this method is closer to the given instructions
        dz2 = A2 - Y
        dw2 = 1 / m * np.matmul(dz2, A1.T)
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        # print(np.multiply(A1, (1 - A1)))
        # print(A1 * (1 - A1))
        # equivalent
        dz1 = np.matmul(self.W2.T, dz2) * (np.multiply(A1, (1 - A1)))
        dw1 = 1 / m * np.matmul(dz1, X.T)
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

        self.__W1 = self.W1 - alpha * (dw1)
        self.__b1 = self.b1 - alpha * (db1)
        self.__W2 = self.W2 - alpha * (dw2)
        self.__b2 = self.b2 - alpha * (db2)

        return
        """ this is my alternate method through my own derivation
            works for dw1, dw2, db2, but i couldn't get db1 working
            before switching to above method
            left here because it is an interesting alternative for
            calculating dw1
        """
        dw1 = np.matmul(X, (A1 - Y).T) / m
        db1 = np.sum((A1 - Y), axis=1, keepdims=True) / m
        # print(db1 is what's wrong)
        # print("-" * 45)
        # print(db1)
        dw2 = 1 / m * np.matmul((A2 - Y), A1.T)
        db2 = np.sum((A2 - Y), axis=1, keepdims=True) / m

        self.__W1 = self.W1 - alpha * (dw1.T)
        self.__b1 = self.b1 - alpha * (db1)
        self.__W2 = self.W2 - alpha * (dw2)
        self.__b2 = self.b2 - alpha * (db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neural network
            X: np.ndarray (nx, m) with input data
            Y: np.ndarray (1, m) with correct labels for input data
                nx: number of input features
                m: number of examples
            iterations: positive int - number if iterations to train
            alpha: positive float - learning rate
            updates __W1, __b1, __A1, __W2, __b2, __A2
            returns evaluation of training data after iterations
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        return self.evaluate(X, Y)
