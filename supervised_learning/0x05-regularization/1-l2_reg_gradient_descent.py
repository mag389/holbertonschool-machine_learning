#!/usr/bin/env python3
""" grad descent with l2 reg """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates weiights and biases of nn with l2 grad descent
        Y: one-hot np.ndarray (classes, m) oof correct labels
            classes: num classes, m: num examples
        weights: dict of weights and biases of the nn
        cache dict of outputs of each layer of nn
        alpha: learning rate
        lambtha: l2 reg parameter
        L: number of layers
        uses tanh activation on each layer except the last
            last uses softmax
    """
    m = len(Y[0])
    dzh = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dwh = 1 / m * np.matmul(dzh, cache["A"+str(i - 1)].T)
        dbh = 1 / m * np.sum(dzh, axis=1, keepdims=True)
        dzl = np.matmul(weights["W" + str(i)].T, dzh)
        extra = (1 - np.square((cache["A" + str(i - 1)])))
        dzl = dzl * extra
        l2 = lambtha / m * (weights["W" + str(i)])
        dwh += l2
        weights["W" + str(i)] -= alpha * dwh
        weights["b" + str(i)] -= alpha * dbh
        dzh = dzl
