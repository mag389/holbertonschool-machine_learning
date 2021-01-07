#!/usr/bin/env python3
""" creates the train op """


import tensorflow as tf


def create_train_op(loss, alpha):
    """ that creates the training operation for the network
        loss is the loss of the network’s prediction
        alpha is the learning rate
        Returns: an operation that trains the network using gradient descent
    """
    gdo = tf.train.GradientDescentOptimizer(alpha)
    grads = gdo.compute_gradients(loss)
    op = gdo.apply_gradients(grads)
    return op
