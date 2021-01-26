#!/usr/bin/env python3
""" prediction from a model """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ predicts based on the data given using the network """
    return network.predict(data, verbose=verbose)
