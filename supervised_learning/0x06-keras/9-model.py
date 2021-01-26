#!/usr/bin/env python3
""" save and load model functions """
import tensorflow.keras as K


def save_model(network, filename):
    """ save the netwoork model to filename. """
    network.save(filename)
    return None


def load_model(filename):
    """ load a network model from a file """
    network = K.models.load_model(filename)
    return network
