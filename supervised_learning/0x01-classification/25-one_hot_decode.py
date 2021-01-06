#!/usr/bin/env python3
""" one hot decode file """
import numpy as np


def one_hot_decode(one_hot):
    """ decodes the one hot matrix """
    try:
        decoded = np.argmax(one_hot, axis=0)
        if decoded.shape != ((one_hot.shape[1],)):
            return None
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
