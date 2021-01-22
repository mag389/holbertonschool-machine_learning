#!/usr/bin/env python3
""" forward prop with dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ conducts forward prop using dropout
        X: np.ndarray (nx, m) of input data
            nx: input features, m: data points
        weights: dictionary of weights and biases
        L: number of layers
        keep_prob: probability the ndoe will be kept
        all layers but last use tanh activation
        last uses softmax
        Returns: dictionary of outputs for each layer
            and dropout mask used on each layer
    """
    cache = {"A0": X}
    for i in range(L):
        z = (np.matmul(weights["W" + str(i + 1)], cache["A" + str(i)])
             + weights["b" + str(i + 1)])
        # create drop layer
        drop = np.random.binomial(1, keep_prob, size=z.shape)
        # apply softmax
        if i == L - 1:
            t = np.exp(z)
            cache["A" + str(i + 1)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache["A" + str(i + 1)] = np.tanh(z)
            cache["D" + str(i + 1)] = drop
            cache["A" + str(i + 1)] = (cache["A" + str(i + 1)] *
                                       cache["D" + str(i + 1)] / keep_prob)
        return cache
