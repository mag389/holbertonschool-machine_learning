#!/usr/bin/env python3
""" creates inception networks """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds the inception network using previous inception block function"""
    data_in = K.Input((224, 224, 3))
    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                            activation='relu', padding='same')(data_in)
    mpool1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                padding='same')(conv1)
    conv2 = K.layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1),
                            activation='relu', padding='same')(mpool1)
    conv3 = K.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1),
                            activation='relu', padding='same')(conv2)
    mpool2 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                padding='same')(conv3)

    i1 = inception_block(mpool2, [64, 96, 128, 16, 32, 32])
    i2 = inception_block(i1, [128, 128, 192, 32, 96, 64])
    mpool3 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                padding='same')(i2)
    i3 = inception_block(mpool3, [192, 96, 208, 16, 48, 64])
    i4 = inception_block(i3, [160, 112, 224, 24, 64, 64])
    i5 = inception_block(i4, [128, 128, 256, 24, 64, 64])
    i6 = inception_block(i5, [112, 144, 288, 32, 64, 64])
    i7 = inception_block(i6, [256, 160, 320, 32, 128, 128])
    mpool4 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                padding='same')(i7)

    i8 = inception_block(mpool4, [256, 160, 320, 32, 128, 128])
    i9 = inception_block(i8, [384, 192, 384, 48, 128, 128])
    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                        padding='valid')(i9)
    # pool should be valid not same based on output sizes
    dropl = K.layers.Dropout(rate=0.4)(avgpool)
    linear = K.layers.Dense(1000, activation='softmax')(dropl)
    model = K.Model(inputs=data_in, outputs=linear)
    return model
