#!/usr/bin/env python3
""" calculating posterior, left side of bayes
    aka prob of developing sideeffects given the data
    This time with continuous distributions
"""
from scipy import special


def posterior(x, n, p1, p2):
    """ calculates posterior probability: the probability of ending up in x
          (With severe side-effects) given you are in n (taking the drug)
        x: number of patients that develop sideeffects
        n: total patients
        p1: lower bound of the range
        p2: upper bound of range
        Returns: posterior prob that p is in range [p1, p2] given x and n
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        error = "x must be an integer that is greater than or equal to 0"
        raise ValueError(error)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise TypeError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise TypeError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("P2 must be greater than p1")
    choose = special.comb(n, x)
    # for binomial distribution conjugate prior is beta (parameters a, b)
    # so doing bayesian flip p(p|x, n) is alos beta
    # we can use scipy to easily cal beta values
    upper = special.btdtr(1 + x, 1 + n - x, p2)
    lower = special.btdtr(1 + x, 1 + n - x, p1)
    return upper - lower
    # choose = np.math.factorial(n) / np.math.factorial(x)
    # choose /= np.math.factorial(nk)
    for i in range(len(P)):
        R[i] *= (P[i]**x) * ((1-P[i])**nk)
    inter = R * Pr
    # R is p(B|A) pr is p(A)
    # so p(b) = sum(p(B|A) *p(A) for all A, which is the whole list
    marginal = np.sum(inter)
    # bayes: p(A|B) = P(B|A) * P(A) / P(B)
    return inter / marginal
