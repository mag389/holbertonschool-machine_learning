#!/usr/bin/env python3
""" integrates a polynomial"""


def poly_integral(poly, C=0):
    """integrate the polynomial poly, return list of co-efficients"""
    if not poly:
        return None
    if not isinstance(C, int) and not isinstance(C, float):
        return None
    if not isinstance(poly, list):
        return None
    if len(poly) <= 0:
        return None
    retval = [0 for x in range(len(poly) + 1)]
    for i in range(len(poly)):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        retval[i + 1] = poly[i] / (i + 1)
        if retval[i + 1] == int(retval[i + 1]):
            retval[i + 1] = int(retval[i + 1])
    retval[0] = C
    if int(C) == C:
        retval[0] == int(C)
    for i in range(len(retval)):
        if retval[len(retval) - i - 1] == 0:
            retval.pop(len(retval) - i - 1)
        else:
            break
    return retval
