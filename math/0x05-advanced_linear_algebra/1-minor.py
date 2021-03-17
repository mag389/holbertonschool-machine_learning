#!/usr/bin/env python3
""" create minor  matrix """


def minor(matrix):
    """ calcs minor matrix of matrix"""
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
    if type(matrix) is not list or len(matrix) < 1:
        raise TypeError("matrix must be a list of lists")
    size = len(matrix)
    # print(size)
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        # print(len(row))
        if len(row) != size:
            raise TypeError("matrix must be a square matrix")
    if size == 1:
        return matrix[0][0]
    num = deter(matrix)
    if num >= 0:
        return int(num + .5)
    return int(num - 0.5)


def deter(matrix):
    """ uses gaussian elimination steps """
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
        coeff /= 1 / matrix[0][0]
        matrix[0] = [matrix[0][x] / matrix[0][0] for x in range(size)]
        # print("cur matr")
        # print(matrix)
        for i in range(1, size):
            old = matrix[i][0]
            for j in range(size):
                # old = matrix[i][0]
                # print("old is: %i", old)
                matrix[i][j] = matrix[i][j] - matrix[0][j] * old
        # print(matrix)
        slicer = [matrix[x][1:] for x in range(1, len(matrix))]
        # print(slicer)
        # print(coeff)
        # print("--------------")
        return (coeff * deter(slicer))
