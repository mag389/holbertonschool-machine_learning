#!/usr/bin/env python3
""" on-hot encode function file """


import numpy as np


def one_hot_encode(Y, classes):
    """one hot encode the data Y into classes number of classes """
    try:
        encoded = np.eye(classes)[Y].T
        if encoded.shape != (classes, Y.shape[0]):
            return None
        return np.eye(classes)[Y].T
    except Exception:
        return None
