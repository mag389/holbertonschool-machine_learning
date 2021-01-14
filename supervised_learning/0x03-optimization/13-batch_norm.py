#!/usr/bin/env python3
""" batch normalizatoin function """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes unactivated output of a neural network using batch norm
        Z: np.ndarray (m, n) that hosuld be normalized
            m data points
            n features in Z
        gamma np.ndarray (1, n) scales for batch normalizaton
        beta: np.ndarray (1, n) offsets used for batch normalizing
            gamma and beta are learnable model params, update through train
        epsilon: small number to not divide by 0
        Returns: normalized matrix Z
    """
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    z_norm = (Z - mean) / np.sqrt(std**2 + epsilon)
    return z_norm * gamma + beta
