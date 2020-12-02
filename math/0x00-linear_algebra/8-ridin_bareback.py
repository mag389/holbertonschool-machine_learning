#!/usr/bin/env python3
""" a basic matrix multiply function """


def mat_mul(mat1, mat2):
    """ multiples matrix one and matrix 2 """
    if len(mat1[0]) != len(mat2):
        return None
    newmat = [[0 for x in range(len(mat2[0]))] for y in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            newnum = 0
            for k in range(len(mat1[0])):
                newnum += mat1[i][k] * mat2[k][j]
            newmat[i][j] = newnum
    return newmat
