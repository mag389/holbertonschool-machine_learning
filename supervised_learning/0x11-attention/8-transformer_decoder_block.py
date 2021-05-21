#!/usr/bin/env python3
""" the decode block class file for transformers """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ the Transformer decoder block class """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Decoder block initializer
            dm: in dimensionality of model
            h: number of heads
            hidden: number of hidden units in FC layer
            drop_rate: dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ the call method for the transformer decoder block
            x: tensor(batch, target_seq_len, dm) of input to decoder block
            encoder_output: tensor(batch, input_seq_len, dm) of encoder output
            training: bool for if model is training
            look_ahead_mask: mask to use for first MHA layer
            padding_mask: mask for second mha layer
            Returns: Tensor(batch, target_seq_len, dm) of block output
        """
        attn1, attm_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            encoder_output, encoder_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # the feed forward part of the network
        out3 = self.dense_hidden(out2)
        out4 = self.dense_output(out3)
        out5 = self.dropout3(out4, training=training)
        out6 = self.layernorm3(out5 + out2)

        return output3
