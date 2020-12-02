#!/usr/bin/env python3
"""take a multidmensional slice of a matrix """
import numpy as np


def np_slice(matrix, axes={}):
    """slice matrix according to slicer dict"""
    slicer = [slice(None) for x in range(len(matrix.shape))]
    for key, value in axes.items():
        slicer[key] = slice(*value)
    return matrix[slicer]
