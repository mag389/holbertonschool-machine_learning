#!/usr/bin/env python3
""" transformer encoder block """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ encoder block class for transformers """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ initializer for transformer encoder
            dm: int: dimensinoality of model
            h: number of heads
            hidden: number of hidden units in fully connected layer
            drop_rate: the dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        """ call method for the transformer encoder block
            x: tensor(batch, input_seq_len, dm) of input to encoder block
            training: bool to determine if model is training
            mask: mask to be applied for mha
            Returns: Tensor(batch, input_seq_len, dm) of block's output
        """
        attn_outputi, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        out2 = self.dense_hidden(out1)
        out3 = self.dense_output(out2)
        out4 = self.dropout2(out3, training=training)
        output = self.layernorm2(ou1 + out4)
        return output
