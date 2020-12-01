#!/usr/bin/env python3


def matrix_shape(matrix):
    """ returns the matrix shape of a given matrix """
    if len(matrix) is 0:
        return [0]
    if type(matrix[0]) is list:
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]
