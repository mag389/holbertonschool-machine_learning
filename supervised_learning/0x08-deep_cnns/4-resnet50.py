#!/usr/bin/env python3
""" resent implementation """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds resNet-50 architecture as in 'Deep Residual Learning for
        Image Recognition" (2015) paper
        Data comes in shape (224, 224, 3)
        convolutions are followed by batch normalizatoin and ReLU activation
        Returns: the keras model
    """
    init = K.initializers.he_normal()
    data_in = K.Input((224, 224, 3))
    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same',
                            kernel_initializer=init)(data_in)
    bn1 = K.layers.BatchNormalization()(conv1)
    relu1 = K.layers.ReLU()(bn1)
    mpool1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                padding='same')(relu1)

    i1 = projection_block(mpool1, [64, 64, 256], s=1)
    i2 = identity_block(i1, [64, 64, 256])
    i3 = identity_block(i2, [64, 64, 256])

    i4 = projection_block(i3, [128, 128, 512])
    i5 = identity_block(i4, [128, 128, 512])
    i6 = identity_block(i5, [128, 128, 512])
    i7 = identity_block(i6, [128, 128, 512])

    i8 = projection_block(i7, [256, 256, 1024])
    i9 = identity_block(i8, [256, 256, 1024])
    i10 = identity_block(i9, [256, 256, 1024])
    i11 = identity_block(i10, [256, 256, 1024])
    i12 = identity_block(i11, [256, 256, 1024])
    i13 = identity_block(i12, [256, 256, 1024])

    i14 = projection_block(i13, [512, 512, 2048])
    i15 = identity_block(i14, [512, 512, 2048])
    i16 = identity_block(i15, [512, 512, 2048])

    avg = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                    padding='valid')(i16)
    linear = K.layers.Dense(1000, activation='softmax')(avg)
    model = K.Model(inputs=data_in, outputs=linear)
    return model
