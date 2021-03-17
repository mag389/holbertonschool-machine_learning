#!/usr/bin/env python3
""" find determinant of a matrix """


def determinant(matrix):
    """ calculate determinant of matrix """
    if matrix == [[]]:
        return 1
    if type(matrix) is not list or len(matrix) < 1:
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    size = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != size:
            raise TypeError("matrix must be a square matrix")
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    num = deter(matrix)
    return round(num)


def deter(matrix):
    """ uses gaussian elimination steps """
    size = len(matrix)
    coeff = 1
    if matrix[0][0] == -500000000:
        for i in range(1, size):
            if matrix[i][0] != 0:
                matrix[i], matrix[0] = matrix[0], matrix[i]
                return -1 * deter(matrix)
        return 0
    else:
        if size == 1:
            return matrix[0][0]
        coeff *= matrix[0][0]
        if coeff == 0:
            coeff = 1.0e-18
        # matrix[0] = [matrix[0][x] / matrix[0][0] for x in range(size)]
        matrix[0] = [matrix[0][x] / coeff for x in range(size)]
        # print("cur matr")
        # print(matrix)
        for i in range(1, size):
            old = matrix[i][0]
            for j in range(size):
                matrix[i][j] = matrix[i][j] - matrix[0][j] * old
        # print(matrix)
        slicer = [matrix[x][1:] for x in range(1, len(matrix))]
        # print(slicer)
        # print(coeff)
        # print("--------------")
        return (coeff * deter(slicer))
