#!/usr/bin/env python3
""" learning rate decay in tensorflow """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ creates learning rate decay opertaion in tensorflow
        alpha: original learning rate
        decay_rate: weight for how fast alpha will decay
        global_step: the current gradient descent pass number
        decay_step: how often to decay alpha:
        Returns: learning rate decay operation
    """
    rate = tf.train.inverse_time_decay(alpha, global_step,
                                       decay_step, decay_rate, staircase=True)
    return rate
