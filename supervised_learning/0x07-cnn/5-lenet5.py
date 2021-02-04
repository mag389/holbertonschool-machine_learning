#!/usr/bin/env python3
""" lenet5 but now with keras """
import tensorflow.keras as K


def lenet5(X):
    """ builds modified LeNet-5 using keras
        X: a K.input of shape (m, 28, 28, 1) of images
        l1:Convolutional layer with 6 kernels of shape 5x5 with same padding
        l2:Max pooling layer with kernels of shape 2x2 with 2x2 strides
        l3:Convolutional layer with 16 kernels of shape 5x5 with valid padding
        l4:Max pooling layer with kernels of shape 2x2 with 2x2 strides
        l5:Fully connected layer with 120 nodes
        l6:Fully connected layer with 84 nodes
        l7:Fully connected softmax output layer with 10 nodes
        Return: K.Model compiled to use Adam, and accuracy metrics
    """
    init = K.initializers.he_normal(seed=None)
    l1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                         padding='same', activation='relu',
                         kernel_initializer=init)(X)
    l2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(l1)
    l3 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                         padding='valid', activation='relu',
                         kernel_initializer=init)(l2)
    l4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(l3)
    flat = K.layers.Flatten()(l4)
    l5 = K.layers.Dense(120, activation='relu',
                        kernel_regularizer=init)(flat)
    l6 = K.layers.Dense(84, activation='relu',
                        kernel_regularizer=init)(l5)
    output = K.layers.Dense(10, activation='softmax',
                            kernel_regularizer=init)(l6)

    model = K.models.Model(inputs=X, outputs=output)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
