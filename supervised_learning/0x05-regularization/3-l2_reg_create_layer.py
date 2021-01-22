#!/usr/bin/env python3
""" creates l2 regularization layer """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ creates tensorflow layer that includes l2 regularization
        prev: tensor of output of prev layer
        n: number of nodes for new layer
        activaiton: activation function for ne wlayer
        lambtha: l2 reg param
        returns: output of new layer
    """
    kernel_r = tf.contrib.layers.l2_regularizer(lambtha)
    kernel_i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    newL = tf.layers.Dense(n,
                           activation=activation,
                           kernel_regularizer=kernel_r,
                           kernel_initializer=kernel_i)(prev)
    return newL
