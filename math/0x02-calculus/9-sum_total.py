#!/usr/bin/env python3
"""summation function file"""


def summation_i_squared(n):
    """ return sum from 1 to n of i^2"""
    if not n:
        return None
    if not isinstance(n, int) and not isinstance(n, float):
        return None
    if n <= 0:
        return None
    if n == 1:
        return(1)
    return(n * n + summation_i_squared(n - 1))
