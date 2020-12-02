#!/usr/bin/env python3
""" concatenating 2d matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenate two matrices along given axis
       axis=0 adds new row(s)
       axis=1 adds new comumn(s)
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        height = len(mat1) + len(mat2)
        width = len(mat1[0])
        newmat = [[0 for x in range(width)] for y in range(height)]
        for i in range(len(mat1)):
            for j in range(len(mat1[i])):
                newmat[i][j] = mat1[i][j]
        for i in range(len(mat2)):
            for j in range(len(mat2[i])):
                newmat[i + len(mat1)][j] = mat2[i][j]
        return newmat
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        height = len(mat1)
        width = len(mat1[0]) + len(mat2[0])
        newmat = [[0 for x in range(width)] for y in range(height)]
        for i in range(len(mat1)):
            for j in range(len(mat1[0])):
                newmat[i][j] = mat1[i][j]
        for i in range(len(mat2)):
            for j in range(len(mat2[0])):
                newmat[i][j + len(mat1[0])] = mat2[i][j]
        return newmat
    return None
