#!/usr/bin/env python3
""" rmsprop in tensorflow """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates a training operation for neural network with rmsprop
        loss: netowrk loss
        alpha: learning rate
        beta2: rmsprop weight
        epsilon: small number to avoid division by zero

        unclear if theres performance differences if using this method
        of compute then apply grads, or calling minimize on rms_o
    """
    rms_o = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    grads = rms_o.compute_gradients(loss)
    return rms_o.apply_gradients(grads)
