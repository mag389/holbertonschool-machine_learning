#!/usr/bin/env python3
""" builds modified lenet5 with tensorflow without keras """
import tensorflow as tf


def lenet5(x, y):
    """ builds modified lenet5
        x: tf.placeholder (m, 28, 28, 1) input images (m is number of images)
        y: tf.placeholder (m, 10) one-hot labels
        in order layers:
            l1: conv layer, 6 kernels (5x5) same padding
            l2: max pool layer 2x2 kernel with 2,2 strides
            l3: conv layer 16 kernels (5x5) valid padding
            l4: max pool layer 3x3 kernel with 2,2 strides
            l5: fully connected layer, 120 nodes
            l6: fully connected, layer 84 nodes
            l7: fully connected softmax output layer, 10 nodes
        hidden layers use ReLu
        Returns: activated output tensor, adam training op, loss tensor
          and accurcy tensor
    """
    kinit = tf.contrib.layers.variance_scaling_initializer()
    acti = tf.nn.relu

    l1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                          padding='same',
                          activation=acti,
                          kernel_initializer=kinit)(x)

    l2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(l1)

    l3 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                          padding='valid',
                          activation=acti,
                          kernel_initializer=kinit)(l2)

    l4 = tf.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2))(l3)

    flat = tf.layers.Flatten()(l4)

    l5 = tf.layers.Dense(units=120, activation=acti,
                         kernel_initializer=kinit)(flat)

    l6 = tf.layers.Dense(units=84, activation=acti,
                         kernel_initializer=kinit)(l5)

    l7 = tf.layers.Dense(units=10,
                         kernel_initializer=kinit)(l6)

    y_pred = tf.nn.softmax(l7)

    loss = tf.losses.softmax_cross_entropy(y, l7)

    train = tf.train.AdamOptimizer().minimize(loss)

    equality = tf.equal(tf.argmax(y, axis=1),
                        tf.argmax(l7, axis=1))
    acc = tf.reduce_mean(tf.cast(diff, tf.float32))

    return y_pred, train, loss, acc
