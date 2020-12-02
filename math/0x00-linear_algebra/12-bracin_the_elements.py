#!/usr/bin/env python3
""" element wise matrix arithmetic functions"""


def np_elementwise(mat1, mat2):
    """ return a tuple of the +, -, *, / of matrix1 one and matrix2 """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
