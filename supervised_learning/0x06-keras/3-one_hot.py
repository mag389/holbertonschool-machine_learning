#!/usr/bin/env python3
""" keras one-hot """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ creates one_hot matrix """
    return K.utils.to_categorical(labels, classes)
