#!/usr/bin/env python3
""" projection block creation """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ builds a projection block as in 'deep residual learning for image
          Recognition' paper
        A_prev: output form previous layer
        filters: tuple or list of (respectively)
            F11: number of filters in first 1x1 conv
            F3: number of filters in 3x3 conv
            F12: number of filters in second 1x1 conv
        s: stride for first convolution and shortcut
        conv's use relu activation and batch norm's
        weights use he normal initialization
        Returns: activated output of identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(F11, kernel_size=(1, 1),
                            padding='same',
                            strides=(s, s),
                            kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization()(conv1)
    relu1 = K.layers.ReLU()(bn1)

    conv2 = K.layers.Conv2D(F3, kernel_size=(3, 3), padding='same',
                            kernel_initializer=init)(relu1)
    bn2 = K.layers.BatchNormalization()(conv2)
    relu2 = K.layers.ReLU()(bn2)

    conv3 = K.layers.Conv2D(F12, kernel_size=(1, 1), padding='same',
                            kernel_initializer=init)(relu2)
    bn3 = K.layers.BatchNormalization()(conv3)

    shortcut = K.layers.Conv2D(F12, kernel_size=(1, 1),
                               padding='same',
                               strides=(s, s),
                               kernel_initializer=init)(A_prev)
    bns = K.layers.BatchNormalization()(shortcut)
    layersum = K.layers.Add()([bn3, bns])
    output = K.layers.ReLU()(layersum)
    return output
