#!/usr/bin/env python3
""" add two matrices of unknown size """


def add_matrices(mat1, mat2):
    """ adds matrix one and two together """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if type(mat1[0]) is int or type(mat1[0]) is float:
        newmat = [0 for x in range(len(mat1))]
        for i in range(len(mat1)):
            newmat[i] = mat1[i] + mat2[i]
        return newmat
    else:
        newmat = [0 for x in range(len(mat1))]
        for i in range(len(mat1)):
            newmat[i] = add_matrices(mat1[i], mat2[i])
        return newmat


def matrix_shape(matrix):
    """ returns the matrix shape of a given matrix """
    if len(matrix) is 0:
        return [0]
    if type(matrix[0]) is list:
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]
