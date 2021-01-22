#!/usr/bin/env python3
""" gradient descent with dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates weights with dropout regularization
        Y: np.ndarray (classes, m) one hot of correct labels
        weights: dict of weights and biases
        cache: dict of output and dropout masks of each layer
        alpha: learning rate
        keep_prob: probabililty a node will be kept
        L: number of layers
        all layers but last use tanh
        update weights in place
    """
    m = len(Y[0])
    dzh = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dwh = 1 / m * np.matmul(dzh, cache["A" + str(i - 1)].T)
        dbh = 1 / m * np.sum(dzh, axis=1, keepdims=True)
        dzl = np.matmul(weights["W" + str(i)].T, dzh)
        extra = 1 - np.square((cache["A" + str(i - 1)]))
        dzl = dzl * extra
        if i != 1:
            dzl = dzl * cache["D" + str(i - 1)] / keep_prob
        # dwh = dwh * cache["D" + str(i - 1)]
        # dbh = dbh * cache["D" + str(i - 1)]
        weights["W" + str(i)] -= alpha * dwh
        weights["b" + str(i)] -= alpha * dbh
        dzh = dzl
