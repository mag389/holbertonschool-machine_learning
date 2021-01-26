#!/usr/bin/env python3
""" sets ups adam opt for keras with crossentropy loss and accuracy metrics """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ set up adam optimization
        network: the model to optimize
        alpha: learning rate
        beta1: first adam paramter
        beta2: second adam param
        Returns: None
    """
    a_o = K.optimizers.Adam(alpha, beta1, beta2)
    # accuracy = K.metrics.CategoricalAccuracy()
    # apparently that line fails
    # that might just retrieve accuracy but the compile function may take
    # metrics=['accuracy'] and loss='categorical_crossentropy'
    # https://www.programcreek.com/python/example/104282/keras.optimizers.Adam
    # loss = K.losses.CategoricalCrossentropy(from_logits=True)
    # another failed line
    # network.compile(loss=loss, optimizer=a_o, metrics=[acuracy])
    network.compile(loss='categorical_crossentropy',
                    optimizer=a_o, metrics=['accuracy'])
    return None
