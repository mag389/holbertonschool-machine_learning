#!/usr/bin/env python3
""" learning rate decay algo """
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates the learning rate using invere time decay in numpy
        alpha: original learning rate
        decay_rate: weight used to determine rate alpha will deacy
        global_step: number of passes gradient descent has elapsed
        decay_step: number of passes of gradient descent between decay steps
        learning rate decays in stepwise fashion
        Returns:updated alpha value
    """
    alpha = alpha * 1 / (1 + decay_rate * (global_step // decay_step))
    return alpha
