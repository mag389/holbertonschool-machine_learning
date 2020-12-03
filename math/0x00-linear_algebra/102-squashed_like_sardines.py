#!/usr/bin/env python3
"""concatenating matrices for comparing to np """


def cat_matrices(mat1, mat2, axis=0):
    """concatenate two matrices along the given axis """
    mat1s = matrix_shape(mat1)
    mat2s = matrix_shape(mat2)
    # print(mat1s, mat2s, axis)
    if len(mat1s) != len(mat2s):
        return None
    for i in range(len(mat1s)):
        if i == axis:
            continue
        if mat1s[i] != mat2s[i]:
            return None
    if axis == 0:
        # if type(mat1[0]) != type(mat2[0]):
        # deprecated method
        if not isinstance(mat1[0], type(mat2[0])):
            return None
        if len(matrix_shape(mat1)) != len(matrix_shape(mat2)):
            return None
        newmat = [0 for x in range(len(mat1) + len(mat2))]
        if not isinstance(mat1[0], list) or not isinstance(mat2[0], list):
            for i in range(len(mat1)):
                newmat[i] = mat1[i]
            for i in range(len(mat2)):
                newmat[i + len(mat1)] = mat2[i]
        else:
            for i in range(len(mat1)):
                newmat[i] = deepercopy(mat1[i])
            for i in range(len(mat2)):
                newmat[i + len(mat1)] = deepercopy(mat2[i])
        return newmat

    if axis >= 1:
        if len(matrix_shape(mat1)) <= axis or len(matrix_shape(mat2)) <= axis:
            return None
        newmat = [0 for x in range(len(mat1))]
        for i in range(len(mat1)):
            if not isinstance(mat1[i], type(mat2[i])):
                return None
            newmat[i] = cat_matrices(mat1[i], mat2[i], axis - 1)
            if newmat[i] is None:
                return None
        return newmat
    return None


def matrix_shape(matrix):
    """ returns the matrix shape of a given matrix """
    if not matrix:
        return None
    if len(matrix) is 0:
        return [0]
    if type(matrix[0]) is list:
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]


def deepercopy(matrix):
    """deep copy a matrix of ints or floats"""
    if isinstance(matrix[0], int) or isinstance(matrix[0], float):
        newmat = [0 for x in range(len(matrix))]
        for i in range(len(matrix)):
            newmat[i] = matrix[i]
        return newmat
    else:
        newmat = [0 for x in range(len(matrix))]
        for i in range(len(matrix)):
            newmat[i] = deepercopy(matrix[i])
        return newmat
