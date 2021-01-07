#!/usr/bin/env python3
""" create placeholders function file """


import tensorflow as tf


def create_placeholders(nx, classes):
    """returns two placeholders for a neural network
       nx: number of feature columns in data
       classes: number of classes in classifier
       returns: x, y
           x: placeholder for input data to neural network
           y: placeholder for one-hot labels for input data
    """
    x = tf.placeholder("float", (None, nx), "x")
    y = tf.placeholder("float", (None, classes), "y")
    return x, y
