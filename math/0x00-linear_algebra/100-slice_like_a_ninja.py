#!/usr/bin/env python3
"""take a multidmensional slice of a matrix """


def np_slice(matrix, axes={}):
    """slice matrix according to slicer dict"""
    slicer = [slice(None) for x in range(len(matrix.shape))]
    # #print(slicer)
    for key, value in axes.items():
        # #args = [None, None, None]
        # #for elem in range(len(value)):
        # #    args[elem] = value[elem]
        # #print(args)
        slicer[key] = slice(*value)
    return matrix[slicer]
