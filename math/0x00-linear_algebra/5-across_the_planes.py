#!/usr/bin/env python3
""" add matrix by element"""


def add_matrices2D(mat1, mat2):
    """adds the elements for two matrices, assume same type
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    newmat = [[0 for x in range(len(mat1))] for y in range(len(mat1[0]))]
    for i in range(len(mat1)):
        for j in range(len(mat1[i])):
            newmat[i][j] = mat1[i][j] + mat2[i][j]
    return newmat
