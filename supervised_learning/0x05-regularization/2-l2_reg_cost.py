#!/usr/bin/env python3
""" l2 regularized cost in tensorflow """
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    """ calcs cost of nn with l2 regularization
        cost: tensor of cost without l2 reg
        Returns: tensor of cost with l2
    """
    # regularizer = tf.contrib.layers.l2_regularizer(0.0)
    # regularizer = tf.nn.l2_loss(cost)
    regularizer = tf.losses.get_regularization_losses()
    # cost = tf.losses.get_losses() this does actually give the costs
    # return cost
    return cost + regularizer
