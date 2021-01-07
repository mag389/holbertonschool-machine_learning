#!/usr/bin/env python3
""" layer creation file """


import tensorflow as tf


def create_layer(prev, n, activation):
    """ creates layer for neurl network:
        prev: tensor output from previous layer
        n: number of nodes in the layer to create
        activation: the activatoin f'n that the layer hsould use
        layers initialized with he et. al
        each layer named layer
        returns tensor output of the layer
    """
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
                                                  mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=kernel_initializer,
                            name="layer")
    return layer(prev)
