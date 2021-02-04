#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import tensorflow.keras as K


def lenet5(X):
    """LeNet-5 architecture using tf. 3D image, RGB image - color
    Arg:
       X: K.Input of shape of shape (m, 28, 28, 1)
          containing the input images
    Layers:
       Convolutional layer with 6 kernels of shape 5x5 with same padding
       Max pooling layer with kernels of shape 2x2 with 2x2 strides
       Convolutional layer with 16 kernels of shape 5x5 with valid padding
       Max pooling layer with kernels of shape 2x2 with 2x2 strides
       Fully connected layer with 120 nodes
       Fully connected layer with 84 nodes
       Fully connected softmax output layer with 10 nodes
    Return:
       K.Model comiled Adam optimization and accuracy metrics
    """
    activation = 'relu'
    k_init = K.initializers.he_normal(seed=None)

    layer_1 = K.layers.Conv2D(filters=6, kernel_size=5,
                              padding='same',
                              activation=activation,
                              kernel_initializer=k_init)(X)

    pool_1 = K.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(layer_1)

    layer_2 = K.layers.Conv2D(filters=16, kernel_size=5,
                              padding='valid',
                              activation=activation,
                              kernel_initializer=k_init)(pool_1)

    pool_2 = K.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(layer_2)

    flatten = K.layers.Flatten()(pool_2)

    layer_3 = K.layers.Dense(120, activation=activation,
                             kernel_initializer=k_init)(flatten)

    layer_4 = K.layers.Dense(84, activation=activation,
                             kernel_initializer=k_init)(layer_3)

    output_layer = K.layers.Dense(10, activation='softmax',
                                  kernel_initializer=k_init)(layer_4)

    model = K.models.Model(X, output_layer)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
