#!/usr/bin/env python3
""" matrix transposing """


def matrix_transpose(matrix):
    """returns new matrix that is transpose of the arg """
    height = len(matrix)
    width = len(matrix[0])
    newmat = [[0 for x in range(height)] for y in range(width)]
    for i in range(width):
        for j in range(height):
            newmat[i][j] = matrix[j][i]
    return newmat
