#!/usr/bin/env python3
""" concatenate arrays """


def cat_arrays(arr1, arr2):
    """ concatenate the two arrays arr1 first"""
    newarr = [0 for i in range(len(arr1) + len(arr2))]
    for i in range(len(arr1)):
        newarr[i] = arr1[i]
    for i in range(len(arr2)):
        newarr[i + len(arr1)] = arr2[i]
    return newarr
