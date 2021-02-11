#!/usr/bin/env python3
""" creation of dense block """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ builds dense block for DenseNet
        X: output of prev layer
        nb_filters: an int representing the num of filters in X
        growth_rate: growth rte for the dense block
        layers: the number of layers in the dense block
        uses bottleneck layers use for DenseNet-B
        Returns: the concatenated output of each layer within dense block
            and number of filters in the concat outputs, respectively
    """
    init = K.initializers.he_normal()
    inputs = X
    for i in range(layers):
        # print("into the first iter -------------")
        norms = K.layers.BatchNormalization()(inputs)
        acts = K.layers.ReLU()(norms)
        convs = K.layers.Conv2D(128, kernel_size=(1, 1),
                                kernel_initializer=init,
                                padding='same')(acts)
        norm2 = K.layers.BatchNormalization()(convs)
        acts2 = K.layers.ReLU()(norm2)
        conv2 = K.layers.Conv2D(growth_rate, kernel_size=(3, 3),
                                kernel_initializer=init,
                                padding='same')(acts2)
        # though both function exist this must be lowercase c concat f'n
        concat = K.layers.concatenate([inputs, conv2])
        inputs = concat
        nb_filters += growth_rate
    return inputs, nb_filters
