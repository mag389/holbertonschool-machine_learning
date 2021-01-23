#!/usr/bin/env python3
""" create layer with dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ creates nn layer with dropout
        prev: tensor with output of prev layer
        n: number of nodes
        activation: activation function that should be used
        keep_prob: probability a node will be kept
        returns: output of new layer
    """
    kernel_i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    newL = tf.layers.Dense(n,
                           activation=activation,
                           kernel_initializer=kernel_i)
    dropper = tf.layers.Dropout(keep_prob)
    # dropper applies to the layer, not the data
    return dropper(newL(prev))
