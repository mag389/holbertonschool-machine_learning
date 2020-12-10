#!/usr/bin/env python3
""" finds derivates of polynomial"""


def poly_derivative(poly):
    """returns list of coefficients of the derivative """
    if not poly:
        return None
    if not isinstance(poly, list):
        return None
    if len(poly) <= 0:
        return None
    if len(poly) == 1:
        return [0]
    retval = [0 for x in range(len(poly) - 1)]
    for i in range(len(poly) - 1):
        retval[i] = poly[i + 1] * (i + 1)
    return retval
