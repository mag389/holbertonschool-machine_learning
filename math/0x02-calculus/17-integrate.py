#!/usr/bin/env python3
""" integrates a polynomial"""


def poly_integral(poly, C=0):
    """integrate the polynomial poly, return list of co-efficients"""
    if not poly or not isinstance(C, int):
        return None
    if not isinstance(poly, list):
        return None
    if len(poly) <= 0:
        return None
    retval = [0 for x in range(len(poly) + 1)]
    for i in range(len(poly)):
        retval[i + 1] = poly[i] / (i + 1)
    retval[0] = C
    return retval
