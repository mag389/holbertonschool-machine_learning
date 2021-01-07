#!/usr/bin/env python3
""" the forward propagatoin function using tensorflow """


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ creates forward prop graph for the neural network
        x: placeholder for input data
        layer_sizes:list with number of nodes in each layer
        activations: list containing activation functoins for each layer
        returns: prediction of the network in tensor form
    """
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
