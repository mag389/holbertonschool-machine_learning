#!/usr/bin/env python3
""" calc definiteness of a matrix """
import numpy as np


def definiteness(matrix):
    """ calc's definiteness of matrix
        matrix: np.ndarray to calculate definiteness
        returns:string describing definiteness, or None if none apply
          positive or negative definite, positive or negative semi, and indef
        also throws typeeror if matrix is wrong type
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if matrix.shape != matrix.T.shape:
        return None
    if not np.all(np.abs(matrix - matrix.T < 1e-8)):
        return None

    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            return None
    try:
        w, v = np.linalg.eig(matrix)
    except Exception as e:
        return None
    pd = True
    nd = True
    for value in w:
        if value <= 0:
            pd = False
        if value >= 0:
            nd = False
    if pd is True:
        return("Positive definite")
    """
    nd = True
    for value in w:
        if value >= 0:
            nd = False
    """
    if nd is True:
        return("Negative definite")

    psd = True
    nsd = True
    for value in w:
        if value < 0:
            psd = False
        if value > 0:
            nsd = False
    if psd is True:
        return("Positive semi-definite")
    if nsd is True:
        return("Negative semi-definite")
    return("Indefinite")
