#!/usr/bin/env python3
""" DenseNet networks transition layer """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ builds transition layer for densely connected network
        X: output of prev layer
        nb_filters: int of filters of X
        compression: compression factor for transition layer
        uses DenseNet-C compression
        Returns: output of the transition layer, num of filters in the output
    """
    init = K.initializers.he_normal()

    nb_f = int(nb_filters * compression)

    norms = K.layers.BatchNormalization()(X)
    # not sure why activate here will ask
    actis = K.layers.ReLU()(norms)
    convs = K.layers.Conv2D(nb_f, kernel_size=(1, 1),
                            kernel_initializer=init,
                            padding='same')(actis)
    pools = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                      padding='same')(convs)
    return pools, nb_f
