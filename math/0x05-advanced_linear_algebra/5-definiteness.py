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

    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            return None
    try:
        w, v = np.linalg.eig(matrix)
    except Exception as e:
        return None
    pd = True
    for value in w:
        if value <= 0:
            pd = False
    if pd is True:
        return("Positive definite")
    """
    this method was in links section but miscalssifies
    nd = True
    for i in range(1, size):
        det = np.linalg.det(matrix[:i, :i])
        if i % 2 == 1:
            if det >= 0:
                nd = False
        if i % 2 == 0:
            if det <= 0:
                nd = False
    if nd is True:
        return("negative definite")
    """
    nd = True
    for value in w:
        if value >= 0:
            nd = False
    if nd is True:
        return("negative definite")

    if np.linalg.det(matrix) != 0:
        return("indefinite")

    psd = True
    nsd = True
    for value in w:
        if value < 0:
            psd = False
        if value > 0:
            nsd = False
    if psd is True:
        return("positive semi-definite")
    if nsd is True:
        return("negative semi-definite")
    return None
    # print(w)
    # print(v)
    return("hi")
