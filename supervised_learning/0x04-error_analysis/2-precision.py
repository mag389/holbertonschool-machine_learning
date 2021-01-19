#!/usr/bin/env python3
""" precision calculation """
import numpy as np


def precision(confusion):
    """ that calculates the precision for each class in a confusion matrix:
        confusion: np.ndarray confusion matrix (classes, classes)
        Returns: np.ndarray(classes,) of sensitivities
    """
    classes = confusion.shape[0]
    sens = np.zeros(classes)
    for i in range(classes):
        sens[i] = confusion[i][i] / np.sum(confusion.T[i])
    return sens
