#!/usr/bin/env python3
"""concaenates matrics"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates matrices, axis=0 for vertical, 1 for horizontal"""
    if axis == 0:
        return np.vstack((mat1, mat2))
    return np.hstack((mat1, mat2))
