#!/usr/bin/env python3
""" one hot decode file """
import numpy as np


def one_hot_decode(one_hot):
    """ decodes the one hot matrix """
    return np.argmax(one_hot, axis=0)
