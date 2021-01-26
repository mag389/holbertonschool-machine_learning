#!/usr/bin/env python3
""" test the model """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ tests the model with data and labels """
    return network.evaluate(data, labels, verbose=verbose)
