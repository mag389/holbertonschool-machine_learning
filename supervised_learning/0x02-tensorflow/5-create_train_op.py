#!/usr/bin/env python3
""" creates the train op """


import tensorflow as tf


def create_train_op(loss, alpha):
    """ that creates the training operation for the network
        loss is the loss of the network’s prediction
        alpha is the learning rate
        Returns: an operation that trains the network using gradient descent
    """
    """
       the reason i use two functions instead of just minimze is because
       i was getting a weird error, it was not producing output:
       and saying: ValueError: No gradients provided for any variable
       i tried using different functions here before looking it up
       https://github.com/tensorflow/tensorflow/issues/42038
       according to that it was probably an issue with data being passed
       so i kept changing previous files. it turned out to be in 4-calc. file
       i changed the args there from (y_pred, y) to (y, y_pred) and it worked
       the corss entropy f'n was unclear which one went first and the checker
       accepted my initial wrong guess.
       this file actually worked with checker before it worked on my VM.
       potentially a version issue on my machine
    """
    gdo = tf.train.GradientDescentOptimizer(alpha)
    grads = gdo.compute_gradients(loss)
    op = gdo.apply_gradients(grads)
    return op
