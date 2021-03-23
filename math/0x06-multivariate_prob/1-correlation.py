#!/usr/bin/env python3
""" correlation """
import numpy as np


def correlation(C):
    """ calculates a correlation matrix
        C: np.ndarray (d, d) of a covariance matrix
          d: number of dimensions
        Returns: correlation matrix
    """
    if type(C) != np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    d0, d1 = C.shape
    if d0 != d1:
        raise ValueError("C must be a 2D square matrix")
    squares = np.sqrt(np.diag(C))
    inverse = C / np.outer(squares, squares)
    return inverse
