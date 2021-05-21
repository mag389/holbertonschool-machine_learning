#!/usr/bin/env python3
""" multihead attention class implementation """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ class used to perform multihead attention """

    def __init__(self, dm, h):
        """ initializer for MHA class
            dm: int: dimensionality of the model(aka model depth)
            h: integer representing number of heads
              dm is divisible by h
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """ split last dimensions to be able to pass into sdp
            x: what to split, batch_size: shape to keep
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """ calling the multihead attention block
            Q: tensor(batch, seq_len_q, dk) input to generate query matrix
            K: tensor(batch, seq_len_v, dk) to generate key matrix
            V: tensor(batch, seq_len_v, dv) to generate value matrix
            mask: always None
            Returns: output, weights
              output: tensor(..., seq_len_q, dm) sdp attention
              weights: tensor(..., h, seq_len_q, seq_len_v) of attention weight
        """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        attention, weights = sdp_attention(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        c_attn = tf.reshape(attention, (batch_size, -1, self.dm))
        output = self.linear(c_attn)
        return output, weghts
