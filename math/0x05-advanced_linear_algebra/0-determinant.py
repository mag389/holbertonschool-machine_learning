#!/usr/bin/env python3
""" find determinant of a matrix """


def determinant(matrix):
    """ calculate determinant of matrix """
    if matrix == [[]]:
        return 1

    shape_col = [len(row) for row in matrix]
    if (isinstance(matrix, list)) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == col for col in shape_col):
        raise ValueError("matrix must be a square matrix")
    """
    if not isinstance(matrix, list) or len(matrix) <= 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if type(matrix) is not list or len(matrix) < 1:
        raise TypeError("matrix must be a list of lists")
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise TypeError("matrix must be a square matrix")
    """
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    return deter(matrix)


def deter(matrix):
    """ get determinant by using sub determinants
        very slow for lager matrices
    """
    size = len(matrix)
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    sign = 1
    det = 0
    for idx in range(size):
        copy = [row.copy() for row in matrix]
        copy = copy[1:]
        new_size = len(copy)

        for i in range(new_size):
            copy[i] = copy[i][0:idx] + copy[i][idx + 1:]
        sub = deter(copy)
        det += sign * matrix[0][idx] * sub
        sign *= -1
    return det


def deterreduce(matrix):
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
    return coeff


def deter1(matrix):
    """ deprecated mathod produces same error as above """
    size = len(matrix)
    coeff = 1
    if matrix[0][0] == 0:
        for i in range(1, size):
            if matrix[i][0] != 0:
                matrix[i], matrix[0] = matrix[0], matrix[i]
                return -1 * deter(matrix)
        return 0
    else:
        if size == 1:
            return matrix[0][0]
        coeff *= matrix[0][0]
        # for if i don't switch rows, but made no difference
        # if coeff == 0:
        #     coeff = 1.0e-18
        # matrix[0] = [matrix[0][x] / matrix[0][0] for x in range(size)]
        matrix[0] = [matrix[0][x] / coeff for x in range(size)]
        # print("cur matr")
        # print(matrix)
        for i in range(1, size):
            old = matrix[i][0]
            for j in range(size):
                matrix[i][j] = matrix[i][j] - matrix[0][j] * old
        slicer = [matrix[x][1:] for x in range(1, len(matrix))]
        # print(slicer)
        # print(coeff)
        # print("--------------")
        return (coeff * deter(slicer))
