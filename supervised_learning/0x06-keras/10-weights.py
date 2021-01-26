#!/usr/bin/env python3
""" saveing and loading wieghts """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ saves the weights from network in file filename, in format given """
    # weights = K.layers.get_weights()
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """ loads weights from a file to a netowrk"""
    network.load_weights(filename)
    return None
