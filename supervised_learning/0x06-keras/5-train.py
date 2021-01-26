#!/usr/bin/env python3
""" train the new keras model and validate"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
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
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
