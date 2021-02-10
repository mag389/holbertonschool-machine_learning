#!/usr/bin/env python3
""" builds inception block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ function to build inception block
        A_prev: output from prev layer
        filters: tuple or list containing (respectively)
            F1: number of filters in 1x1 convolution
            F3R: number of filters in 1x1 before the 3x3 conv
            F3: number of filters in 3x3 convolution
            F5R: number of filters in 1x1 before the 5x5 conv
            F5: number of filters in 5x5 convolution
            FPP: number of filters in 1x1 after max pooling
        all usr ReLU activation
        Returns: concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    oneconv = K.layers.Conv2D(F1, (1, 1), padding='same',
                              activation='relu')(A_prev)

    oneC3 = K.layers.Conv2D(F3R, (1, 1), padding='same',
                            activation='relu')(A_prev)
    threeC = K.layers.Conv2D(F3, (3, 3), padding='same',
                             activation='relu')(oneC3)

    oneC5 = K.layers.Conv2D(F5R, (1, 1), padding='same',
                            activation='relu')(A_prev)
    fiveC = K.layers.Conv2D(F5, (5, 5), padding='same',
                            activation='relu')(oneC5)

    pool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                              padding='same')(A_prev)
    poolC = K.layers.Conv2D(FPP, (1, 1), padding='same',
                            activation='relu')(pool)

    return K.layers.Concatenate()([oneconv, threeC, fiveC, poolC])

