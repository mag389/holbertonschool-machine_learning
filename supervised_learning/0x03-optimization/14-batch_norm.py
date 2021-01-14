#!/usr/bin/env python3
""" batch normalization with tensorflow """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ creates a batch normalization layer for a network in tf
        prev: activated output of prev layer
        n: number of nodes in the layer to be created
        activation: activation function to use on the output of the layer
        Returns: a tensor of the activated output for the layer
        batch normalization takes in a layer input not an activated output of
            the previous layer.  That's why we make a layer first without
            activation. then normalize, then activate.
    """
    epsilon = 1e-8
    # mean and ar are also for layered output
    # mean, var = tf.nn.moments(prev, axes=[0])

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, kernel_initializer=init)
    # the layer output from previous layer
    new = layer(prev)

    mean, var = tf.nn.moments(new, axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       trainable=True, name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        trainable=True, name='gamma')

    # z = gamma * prev + beta
    bn = tf.nn.batch_normalization(new, mean, var, beta, gamma, epsilon)
    return activation(bn)
