#!/usr/bin/env python3
""" builds nn with keras input """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ biuld nn with keras, uses dropout and L2. innput model
        nx: number of input features
        layers: number of nodes in each layer
        activations: list containing activation f'ns for each layer
        lambtha: L2 regularization parameter
        keep_prob: prob node will be kept for dropout
        Returns: the keras model
    """
    visible = K.Input(shape=(nx,))
    # model.add(K.Input(shape=(nx,)))
    last = K.layers.Dense(layers[0],
                          activation=activations[0],
                          input_shape=(nx,),
                          kernel_regularizer=K.regularizers.l2(lambtha),
                          # name="layer0"
                          )(visible)
    for i in range(1, len(layers)):
        last = K.layers.Dropout(1 - keep_prob)(last)
        last = (K.layers.Dense(layers[i],
                               activation=activations[i],
                               kernel_regularizer=K.regularizers.l2(lambtha),
                               # name="layer" + str(i + 1)
                               )
                )(last)
    return K.Model(inputs=visible, outputs=last)
