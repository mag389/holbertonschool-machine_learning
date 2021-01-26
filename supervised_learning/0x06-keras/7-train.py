#!/usr/bin/env python3
""" train the new keras model and validate, early stop, and learning decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """ train with minibatch grad desc
        network: the model to train
        data: np.ndarray (m, nx) of input data
        labels: one hot np.ndarray (m, classes) of lanels
        batch_size: sixe of mini batch
        epochs: number of passes through data
        verbose: bool determines if output should print
        shuffle:bool of whether to shuffle every epoch
        Returns: History object generated

        validation data: datta to validate with
        early_stopping:bool of whether to stop early
        patience: if stopping early the patience value to use
        learning_rate_decay: bool if decay should be used
            inverse time decay, stepwise, print when updated
        alpha:initial learn rate
        decay_rate: the decay rate
    """
    if early_stopping and validation_data:
        es = [K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]
    else:
        es = []

    def scheduler(epoch, current_rate):
        """ the inverse time decay function
            could be made with keras directly if not using callbacks
        """
        return alpha / (1 + decay_rate * epoch)

    if learning_rate_decay and validation_data:
        es.append(K.callbacks.LearningRateScheduler(scheduler,
                                                    verbose=1))

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=es,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
