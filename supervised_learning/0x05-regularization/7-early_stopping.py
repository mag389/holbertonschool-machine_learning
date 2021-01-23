#!/usr/bin/env python3
""" determines if should stop gradient descent early """
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """stop early when validation cost of nn has not decreased realtive to
       optimal validation cost by more than the threshold over a
       specific patience count
       cost: current validation cost
       opt_cost: lowest recorded validation cost of the nn
       threshold: threshold used for early stoppage
       patience: patience ocunt used for early stoppage
       count: how long threshold has not been met
       return: Bool of if it should be stopped, and new count
    """
    """
    this as the basics of the when to stop early paper
    but not the mehtod we are using in this assignment
    if (cost / opt_cost - 1) > threshold:
        return True, count
    pk = ( cost / count / opt_cost) - 1
    if (cost / opt_cost - 1) / pk > threshold:
        return True, count
    return False, -1
    if cost > threshold:
        return True, count
    if cost / count > threshold:
        return True, count
    if cost > opt_cost and patience < count:
        return True, count
    return false, count
    """
    if cost < opt_cost - threshold:
        return False, 0
    if count >= patience - 1:
        return True, count + 1
    return False, count + 1
