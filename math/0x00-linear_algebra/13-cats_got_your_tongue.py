#!/usr/bin/env python3
"""concaenates matrics"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates matrices, axis=0 for vertical, 1 for horizontal"""
    return np.concatenate((mat1, mat2), axis)
