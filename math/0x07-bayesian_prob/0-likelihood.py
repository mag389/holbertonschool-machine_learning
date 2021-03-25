#!/usr/bin/env python3
"""calculating likelihood """
import numpy as np


def likelihood(x, n, P):
    """ calculates likelihood of obtaining data gien probabilities
        x: number of patients that develop sideeffects
        n: total patients
        P: 1D np.ndarray of various hypothetical probs of deving sideeffects
        Returns: 1d np.ndarray containing likelihood of obtaining data x and n
            for each probability in p respectively
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        error = "x must be an integer that is greater than or equal to 0"
        raise ValueError(error)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    if len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for num in P:
        if num < 0 or num > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    R = np.ones(P.shape)
    nk = n - x
    choose = np.math.factorial(n) / np.math.factorial(x)
    choose /= np.math.factorial(nk)
    R *= choose
    for i in range(len(P)):
        R[i] *= (P[i]**x) * ((1-P[i])**nk)
    return R
