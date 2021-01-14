#!/usr/bin/env python3
""" adam algo in ensorflow """
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ creates tensorflwo adam operation
        loss: loss of the network
        alpha: learning rate
        beta1: weight for first moment
        beta2: weight for second moment
        epsilon: small number to avoid zero division
        Return: adam optimization operation
    """
    a_o = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return a_o.minimize(loss)
