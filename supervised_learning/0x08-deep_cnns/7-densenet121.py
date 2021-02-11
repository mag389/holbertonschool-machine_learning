#!/usr/bin/env python3
""" denset 121 implement """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ builds a DenseNet-121 architecture as described in 2018 paper
        growth_rate: is the growth rate
        compression: cnn's compression factor
        input data will be (224, 224, 3)
        Returns: the keras model
    """
    init = K.initializers.he_normal()
    data_in = K.Input((224, 224, 3))
    # precede covolutin with BN and ReLU
    norms = K.layers.BatchNormalization()(data_in)
    actis = K.layers.ReLU()(norms)

    # conv2d 7x7 stride=2
    convs = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                            kernel_initializer=init, padding='same')(actis)

    # 3x3 max pool, strie=2
    pools = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(convs)

    # dense block 1: 56 filters, 6 layers, growth rate=growth rate for all
    x, y = dense_block(pools, 64, growth_rate, 6)
    # transition block 1: new filters value, and compression, compression
    x, y = transition_layer(x, y, compression)

    # dense block 2: 28 filters, 12 layers
    x, y = dense_block(x, y, growth_rate, 12)
    # transition block 2:
    x, y = transition_layer(x, y, compression)

    # dense block 3
    x, y = dense_block(x, y, growth_rate, 24)
    # transition block 3
    x, y, = transition_layer(x, y, compression)

    # dense block 4
    x, y = dense_block(x, y, growth_rate, 16)

    # 7x7 avg pool
    avgs = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                     padding='valid')(x)
    # 1000 unit FC classification layer
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(avgs)
    return K.Model(inputs=data_in, outputs=output)
