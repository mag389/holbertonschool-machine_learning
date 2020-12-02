#!/usr/bin/env python3
""" add matrix by element"""


def add_matrices2D(mat1, mat2):
    """adds the elements for two matrices, assume same type
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    newmat = [[0 for x in range(len(mat1))] for y in range(len(mat1[0]))]
    for i in range(len(mat1)):
        for j in range(len(mat1[i])):
            newmat[i][j] = mat1[i][j] + mat2[i][j]
    return newmat


def matrix_shape(matrix):
    """ returns the matrix shape of a given matrix """
    if len(matrix) is 0:
        return [0]
    if type(matrix[0]) is list:
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]
