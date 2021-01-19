#!/usr/bin/env python3
""" sensitivity calculation """
import numpy as np


def sensitivity(confusion):
    """ that calculates the sensitivity for each class in a confusion matrix:
        confusion: np.ndarray confusion matrix (classes, classes)
        Returns: np.ndarray(classes,) of sensitivities
    """
    classes = confusion.shape[0]
    sens = np.zeros(classes)
    for i in range(classes):
        sens[i] = confusion[i][i] / np.sum(confusion[i])
    return sens
