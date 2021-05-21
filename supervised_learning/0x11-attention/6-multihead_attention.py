#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class perform multi-head attention
    """
    def __init__(self, dm, h) -> None:
        """Initializer
        Arguments:
            dm {int} -- Is representing the dimensionality of the model
            h {int} -- Is representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = self.dm // self.h
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

    def call(self, Q, K, V, mask):
        """Instance call
        Arguments:
            Q {tf.Tensor} -- Is a tensor of shape (batch, seq_len, dk) contains
            input to generate query matrix
            K {tf.Tensor} -- Is a tensor of shape (batch, seq_len, dk) contains
            input to generate the key matrix
            V {tf.Tensor} -- is a tensor of shape (batch, seq_len, dv) contains
            input to genereate the value matrix
            mask {[type]} -- None
        Returns:
            tuple -- Contains tf.Tensor of scaled dot product attention, and
            tf.Tensor of the attention weights
        """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        def split_heads(x):
            """Splits inputs into heads
            Arguments:
                x {tf.Tensor} -- An input tensor
            Returns:
                tf.Tensor -- Splited tensor
            """
            x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))
        output = self.linear(output)
        return output, weights
