#!/usr/bin/env python3
""" accuracy function file """


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates accuracy of a prediction
        Return: the accuracy as tensor("Mean:0", shape=(), dtype=float32)
            thattensor is the decimal accuracy of the prediction
    """
    # print(y)
    # print(y_pred)
    y1 = tf.argmax(y, 1)
    yp1 = tf.argmax(y_pred, 1)

    # print("-" * 50)
    # print(y1)
    # print(yp1)
    # print("-" * 50)

    # err = tf.cast(y1 - yp1, tf.float32)
    # acc = tf.reduce_mean(err)
    # print(acc)

    # https://stackoverflow.com/questions/42607930/
    # how-to-compute-accuracy-of-cnn-in-tensorflow
    # no space in url

    equality = tf.math.equal(yp1, y1)
    acc = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return acc
