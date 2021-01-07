#!/usr/bin/env python3
""" loss function file """


import tensorflow as tf


def calculate_loss(y, y_pred):
    """ calculates softmax entropy loss
        Returns a tensor containing loss of the prediction
        i.e.
        Tensor("softmax_cross_entropy_loss/value:0", shape=(), dtype=float32)
    """
    loss = tf.losses.softmax_cross_entropy(y_pred, y)
    return loss
