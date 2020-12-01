#!/usr/bin/env python3
""" add two arrays"""


def add_arrays(arr1, arr2):
    """adds two arrays returns new result """
    if len(arr1) != len(arr2):
        return None
    newlist = [arr1[i] + arr2[i] for i in range(len(arr1))]
    return newlist
