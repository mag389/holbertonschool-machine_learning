#!/usr/bin/env python3
""" train the new keras model """
import tensorflow.keras as K
import numpy as np


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ train with minibatch grad desc
        network: the model to train
        data: np.ndarray (m, nx) of input data
        labels: one hot np.ndarray (m, classes) of lanels
        batch_size: sixe of mini batch
        epochs: number of passes through data
        verbose: bool determines if output should print
        shuffle:bool of whether to shuffle every epoch
        Returns: History object generated
    """
    print("in the method at least")
    if verbose == True:
        verbose = 1
    else:
        verbose = 0
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
