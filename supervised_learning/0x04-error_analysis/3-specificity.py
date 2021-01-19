#!/usr/bin/env python3
""" specificity calculation """
import numpy as np


def specificity(confusion):
    """ that calculates the specificity for each class in a confusion matrix:
        confusion: np.ndarray confusion matrix (classes, classes)
        Returns: np.ndarray(classes,) of sensitivities
    """
    classes = confusion.shape[0]
    spec = np.zeros(classes)
    for i in range(classes):
        negatives = np.sum(confusion) - np.sum(confusion[i])
        true_neg = negatives - np.sum(confusion.T[i]) + confusion[i][i]
        num = true_neg
        false_pos = np.sum(confusion.T[i]) - confusion[i][i]
        denom = true_neg + false_pos
        spec[i] = num / denom
    return spec
