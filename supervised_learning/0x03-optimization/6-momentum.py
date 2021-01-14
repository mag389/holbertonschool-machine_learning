#!/usr/bin/env python3
""" gradient descent with momentum in tensorflow """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ creates training operation for gradient descent with momentum
        loss: the loss of the network
        alpha: the learning rate
        beta1: momentum weight
        returns: the momentum optinixation operation
    """
    mo = tf.train.MomentumOptimizer(alpha, beta1)
    grads = mo.compute_gradients(loss)
    op = mo.apply_gradients(grads)
    return op
