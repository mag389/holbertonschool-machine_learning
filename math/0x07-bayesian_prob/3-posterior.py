#!/usr/bin/env python3
""" calculating posterior, left side of bayes
    aka prob of developing sideeffects given the data
"""
import numpy as np


def posterior(x, n, P, Pr):
    """ calculates posterior probability: the probability of ending up in x
          (With severe side-effects) given you are in n (taking the drug)
        x: number of patients that develop sideeffects
        n: total patients
        P: 1D np.ndarray of various hypothetical probs of deving sideeffects
        Pr: 1d np.ndarray of prior beliefs of P
        Returns: 1d np.ndarray containing likelihood of obtaining data x and n
            for each probability in p respectively
    """
    if type(n) is not int or n <= 0:
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for num in P:
        if num < 0 or num > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for num in Pr:
        if num < 0 or num > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    R = np.ones(P.shape)
    nk = n - x
    choose = np.math.factorial(n) / np.math.factorial(x)
    choose /= np.math.factorial(nk)
    R *= choose
    for i in range(len(P)):
        R[i] *= (P[i]**x) * ((1-P[i])**nk)
    inter = R * Pr
    # R is p(B|A) pr is p(A)
    # so p(b) = sum(p(B|A) *p(A) for all A, which is the whole list
    marginal = np.sum(inter)
    # bayes: p(A|B) = P(B|A) * P(A) / P(B)
    return inter / marginal
