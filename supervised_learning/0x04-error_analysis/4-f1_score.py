#!/usr/bin/env python3
""" f1 calculation """
import numpy as np


def f1_score(confusion):
    """ that calculates the f1_score for each class in a confusion matrix:
        confusion: np.ndarray confusion matrix (classes, classes)
        Returns: np.ndarray(classes,) of sensitivities
    """
    classes = confusion.shape[0]
    prec = np.zeros(classes)
    recall = np.zeros(classes)
    f1 = np.zeros(classes)
    for i in range(classes):
        prec[i] = confusion[i][i] / np.sum(confusion.T[i])
        recall[i] = confusion[i][i] / np.sum(confusion[i])
        f1[i] = 2 / (1 / prec[i] + 1 / recall[i])

    return f1
