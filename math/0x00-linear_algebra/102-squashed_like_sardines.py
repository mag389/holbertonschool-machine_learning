#!/usr/bin/env python3
"""concatenating matrices for comparing to np """


def cat_matrices(mat1, mat2, axis=0):
    """concatenate two matrices along the given axis """
    if axis == 0:
        # if type(mat1[0]) != type(mat2[0]):
        # deprecated method
        if not isinstance(mat1[0], type(mat2[0])):
            return None
        if len(matrix_shape(mat1)) != len(matrix_shape(mat2)):
            return None
        newmat = [0 for x in range(len(mat1) + len(mat2))]
        if not isinstance(newmat[0], list):
            for i in range(len(mat1)):
                newmat[i] = mat1[i]
            for i in range(len(mat2)):
                newmat[i + len(mat1)] = mat2[i]
        else:
            for i in range(len(mat1)):
                newmat[i] = mat1[i].deepcopy()
            for i in range(len(mat2)):
                newmat[i + len(mat1)] = mat2[i].deepcopy()
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
