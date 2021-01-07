#!/usr/bin/env python3
""" the deep neural network class file for defining deep neural network """


import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ the deep neural network class
        a neural network with a variable number of layers
    """

    def __init__(self, nx, layers, activation='sig'):
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
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(1, len(layers) + 1):
            if not isinstance(layers[i - 1], int) or layers[i - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
            """he at al."""
            li = layers[i - 2]
            if i is 1:
                li = nx
            self.__weights["W" + str(i)] = np.random.randn(
                                           layers[i - 1], li) * np.sqrt(2 / li)
            self.__weights["b" + str(i)] = np.zeros((layers[i - 1], 1))

    @property
    def L(self):
        """ getter for number of layers in the deep neural network """
        return self.__L

    @property
    def cache(self):
        """ getter for the dictionary holding intermediate values
            of the network. starts empty.
        """
        return self.__cache

    @property
    def weights(self):
        """ getter for the weights/biases dictionary. weight keys W{layer #}
            and bias keys as 'b{layer number}'
        """
        return self.__weights

    @property
    def activation(self):
        """ getter for activation type """
        return self.__activation

    def forward_prop(self, X):
        """ caculates the forward propagation of the neural network
            X: np.ndarray (nx, m) of input data
                nx: number of input features
                m: number of examples
            updates __cache attribute
            Returns: output of the neural network, and the cache
        """
        self.__cache["A0"] = X.copy()
        for i in range(1, self.L + 1):
            cur_cache = self.cache["A" + str(i - 1)]
            y_tmp = np.matmul(self.weights["W" + str(i)], self.__cache[
                "A" + str(i - 1)]) + self.weights["b" + str(i)]
            Y1 = y_tmp
            if i < self.L:
                if self.activation == 'sig':
                    A_temp = 1 / (1 + np.exp((-1) * Y1))
                else:
                    A_temp = np.tanh(Y1)
                    # A_temp = 2 / (1 + np.exp((-2) * Y1)) - 1
            else:
                # else softmax function
                A_temp = np.exp(Y1) / np.sum(np.exp(Y1), axis=0, keepdims=True)
            self.__cache["A" + str(i)] = A_temp
        return self.cache["A" + str(self.L)], self.cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression
            Y is np.ndarray (classes, m) with correct labels for input data
                it is now a one-hot array
            A is np.ndarray (classes, m) with activted output of data
                m is number of example
            returns: the cost
        """
        m = len(Y[0])
        # J = (-1 / m) * (np.matmul(Y, np.log(A).T) +
        #                np.matmul((1-Y), np.log(1.0000001 - A).T))
        # insteaad of old log diffrence method we now use cross entropy
        J = (-1 / m) * Y * np.log(A)
        return np.sum(J)

    def evaluate(self, X, Y):
        """ evaluates the neural networks predictions
            X: np.ndarray (nx, m) of input data
            Y: np.ndarray (1, m) of correct labels for input data
                nx: number of input features
                m: the number of examples
            Returns: the neuron's prediction and cost (A, cost)
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        # prediction = np.round(A).astype(np.int)
        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ calculates one pass of gradient descent on the neural network
            Y: np.ndarray (1, m) of correct labels for input data
                m: number of examples
            cache: dict of all intermediary values of the network (and input)
            alpha: the learning rate
            Updates: __weights (continas both weight and bias)
        """
        m = len(Y[0])
        dzh = cache["A" + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            dwh = 1 / m * np.matmul(dzh, cache["A"+str(i - 1)].T)
            dbh = 1 / m * np.sum(dzh, axis=1, keepdims=True)
            # sig dz
            if self.activation == 'sig':
                dzl = np.matmul(self.weights["W" + str(i)].T, dzh) * (
                      np.multiply(cache["A" + str(i - 1)], (1 -
                                  cache["A" + str(i - 1)])))
            # tanh dz = 1 - tanh^2
            elif self.activation == 'tanh':
                dzl = np.matmul(self.weights["W" + str(i)].T, dzh) * (
                      1 - np.square(np.tanh(cache["A" + str(i - 1)])))
            self.__weights["W" + str(i)] -= alpha * dwh
            self.__weights["b" + str(i)] -= alpha * dbh
            dzh = dzl

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the deep neural network
            X: np.ndarray (nx, m) with input data
            Y: np.ndarray (1, m) with correct labels for input data
                nx: number of input features
                m: number of examples
            iterations: positive int - number if iterations to train
            alpha: positive float - learning rate

            verbose: bool to tell to print training info
            graph: bool defining to graph info about training
            step:how often to graph/print verbose info
            updates __weights and __cache
            returns evaluation of thr training data
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            g_costs = np.array(())
        for i in range(iterations):
            self.forward_prop(X)
            # A, cache = self.forward_prop(X)
            if (verbose or graph) and (i % step is 0 or i is 0):
                cost = self.cost(Y, self.cache["A" + str(self.L)])
                if verbose:
                    print("cost after {} iterations: {}".format(i, cost))
                if graph:
                    g_costs = np.append(g_costs, cost)
            self.gradient_descent(Y, self.cache, alpha)
        if verbose:
            cost = self.cost(Y, self.cache["A" + str(self.L)])
            print("cost after {} iterations: {}".format(iterations, cost))
        if graph:
            g_costs = np.append(g_costs,
                                self.cost(Y, self.cache["A"+str(self.L)]))
            plt.plot(np.arange(0, iterations + step, step), g_costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ saves the instance object to a file in pickle format """
        if filename[len(filename) - 4:] != ".pkl":
            filename = filename + ".pkl"
        fileobj = open(filename, 'wb')
        pickle.dump(self, fileobj)
        fileobj.close()

    def load(filename):
        """ tries to load from pkl file """
        try:
            with open(filename, 'rb') as fileobj:
                new_obj = pickle.load(fileobj)
                fileobj.close()
            return new_obj
        except Exception:
            return None
