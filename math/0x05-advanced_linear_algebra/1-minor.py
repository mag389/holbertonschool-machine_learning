#!/usr/bin/env python3
""" create minor  matrix """


def minor(matrix):
    """ calcs minor matrix of matrix"""
    if matrix == [[]]:
        return 1
    if type(matrix) is not list or len(matrix) < 1:
        raise TypeError("matrix must be a list of lists")
    size = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        # print(len(row))
        if len(row) != size:
            raise TypeError("matrix must be a square matrix")
    if size == 1:
        return 1
    minors = []
    for i in range(size):
        minors.append([])
        for j in range(size):
            minors[i].append(matrix[i][j])
    # print(minors)
    for i in range(size):
        for j in range(size):
            # print("the matrix")
            # print(matrix)
            b = matrix[:i].copy() + matrix[i + 1:].copy()
            # print(matrix)
            # print("+++++++++++++++")
            slicer = [b[x][:j] + b[x][j + 1:] for x in range(len(b))]
            # print("-------")
            # print(slicer)
            # minors[i][j] = slicer.copy()
            # print("before")
            # print(matrix)
            minors[i][j] = determinant(slicer.copy())
            # print("after")
            # print(matrix)
            # print("===========")
    return minors
    mini = slicer.copy()
    for i in range(size):
        for j in range(size):
            mini = determinant(slicer[i][j])
    return mini


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
