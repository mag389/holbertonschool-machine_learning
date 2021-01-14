#!/usr/bin/env python3
""" rmsprop optimization algo """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates the variable using RMSProp
        alpha:learning rate
        beta2: RMSProp weight
        epsilon: a small number to avoid division by zero
        var: np.ndarray of variable tp update (w)
        grad: np.ndarray with gradient of var (dw)
        s: previous second moment of var (dw_prev)
        Return: updated var and new moment respectively
    """
    sdw = beta2 * s + (1 - beta2) * grad * grad
    var = var - alpha * grad / (np.sqrt(sdw) + epsilon)
    return var, sdw
