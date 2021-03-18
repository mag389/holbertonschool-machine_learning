#!/usr/bin/env python3
""" create minor  matrix """


def adjugate(matrix):
    """ calculates adjugate of matrix
        Matrix: given matrix to get adjugate of
        Returns: new adjugate matrix
    """
    cofactors = cofactor(matrix)
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            cofactors[i][j], cofactors[j][i] = cofactors[j][i], cofactors[i][j]
    return cofactors


def cofactor(matrix):
    """ calculates cofactor matrix of original matrix
        returns: new array
    """
    minors = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            coeff = ((i + j) % 2) * -2 + 1
            minors[i][j] *= coeff
    return minors


def minor(matrix):
    """ calcs minor matrix of matrix"""
    cols = [len(row) for row in matrix]
    if (isinstance(matrix, list)) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == col for col in cols):
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    size = len(matrix)

    if size == 1:
        return [[1]]
    minors = []
    for i in range(size):
        minors.append([])
        for j in range(size):
            minors[i].append(matrix[i][j])
    for i in range(size):
        for j in range(size):
            b = matrix[:i].copy() + matrix[i + 1:].copy()
            slicer = [b[x][:j] + b[x][j + 1:] for x in range(len(b))]
            minors[i][j] = determinant(slicer.copy())
    return minors


def determinant(matrix):
    """ calculate determinant of matrix """
    if matrix == [[]]:
        return 1

    cols = [len(row) for row in matrix]
    if (isinstance(matrix, list)) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == col for col in cols):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    return deter(matrix)


def deter(matrix):
    """ uses elimination without recursion """
    size = len(matrix)
    coeff = 1
    for i in range(size):
        if matrix[i][i] == 0:
            switch = 0
            for j in range(i + 1, size):
                if matrix[j][i] != 0:
                    matrix[j], matrix[i] = matrix[i], matrix[j]
                    coeff *= -1
                    switch = 1
                    break
            if switch == 0:
                return 0
        diag = matrix[i][i]
        if diag == 0:
            return 0
        coeff *= matrix[i][i]
        matrix[i] = [matrix[i][x] / diag for x in range(size)]
        for j in range(i + 1, size):
            old = matrix[j][i]
            for k in range(size):
                matrix[j][k] = matrix[j][k] - matrix[i][k] * old
    return round(coeff)
