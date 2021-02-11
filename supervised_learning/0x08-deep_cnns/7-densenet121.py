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
    # conv2d 7x7 stride=2
    
    # 3x3 max pool, strie=2
    # dense block 1: 56 filters, 6 layers, growth rate=growth rate for all
    # transition block 1: new filters value, and compression, compression

    # dense block 2
    # transition block 2

    # dense block 3
    # transition block 3

    # dense block 4

    # 7x7 avg pool
    # 1000 unit FC classification layer
