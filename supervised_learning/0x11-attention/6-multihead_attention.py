#!/usr/bin/env python3
"""
Class Transformer
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ class"""

    def __init__(self, dm, h):
        """
        class constructor
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(self.dm // self.h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def splitHeads(self, m, batch):
        """
        split last dim shape(self.h, self.depth)
        transpose result shape(batch, -1, self.h, self.depth)
        """
        m = tf.reshape(m, (batch, -1, self.h, self.depth))
        return tf.transpose(m, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Public Instance Method
        """
        batch = tf.shape(K)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.splitHeads(Q, batch)
        K = self.splitHeads(K, batch)
        V = self.splitHeads(V, batch)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        output = self.linear(output)

        return output, weights
