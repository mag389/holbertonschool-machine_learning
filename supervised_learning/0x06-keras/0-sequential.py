#!/usr/bin/env python3
""" builds nn with keras """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ biuld nn with keras, uses dropout and L2
        nx: number of input features
        layers: number of nodes in each layer
        activations: list containing activation f'ns for each layer
        lambtha: L2 regularization parameter
        keep_prob: prob node will be kept for dropout
        Returns: the keras model
    """
    model = K.Sequential()
    # model.add(K.Input(shape=(nx,)))
    model.add(K.layers.Dense(layers[0],
                             activation=activations[0],
                             input_shape=(nx,),
                             kernel_regularizer=K.regularizers.l2(lambtha),
                             # name="layer0"
                             )
              )
    # having input shape means we don't need a kernel initializer
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(keep_prob))
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[i],
                  # kernel_initializer=kernel_i,
                                 kernel_regularizer=K.regularizers.l2(lambtha),
                                 name="layer" + str(i + 1))
                  )
    # model((nx,))
    # many examples use this style initialization
    return model
