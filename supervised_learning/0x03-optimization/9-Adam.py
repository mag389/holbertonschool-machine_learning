#!/usr/bin/env python3
""" Adam optimization algo """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ updates variable with adam algo
        alpha: learning rate
        beta1: weight used for first moment
        beta2: weight used for second moment
        epsilon: small number to avoid division by zero
            those are the hyperparameters
        var: np.ndarray variable to be updated (w)
        grad: np.ndarray gradient of var (dw)
        v: previous first moment (dw_prev1)
        s: previous second moment (dw_prev2)
        t: time step used for bias correction
        Return: updated var, first moment, second moment
    """
    vdw = beta1 * v + (1 - beta1) * grad
    sdw = beta2 * s + (1 - beta2) * grad * grad
    v_corrected = vdw / (1 - beta1**t)
    s_corrected = sdw / (1 - beta2**t)
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, vdw, sdw
