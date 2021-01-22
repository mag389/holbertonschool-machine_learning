#!/usr/bin/env python3
""" l2 regularization cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calcs cost of neural network with l2 regularization
        cost: cist without L2 reg
        lambtha: reg parameter
        weights: dic of weights and biases of the nn
        L number of layers
        m: number of data points used
    """
    coeff = lambtha / 2 / m
    weightsum = 0
    for key, value in weights.items():
        # weightsum += np.sqrt(np.sum(value * value))
        weightsum += np.linalg.norm(value, "fro")
        # aka frobenius norm i.e. += np.linalg.norm(value, "fro")
    return cost + coeff * (weightsum)
