#!/usr/bin/env python3
""" transformer decoder class file """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ the decoder transformer layer class """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """ decoder initialization
            N: blocks in the encoder
            dm - the dimensionality of the model
            h - the number of heads
            hidden - number of hidden units in fully connected layer
            target_vocab - the size of the target vocabulary
            max_seq_len - the maximum sequence length possible
            drop_rate - the dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for i in range(N):
            self.blocks.append(DecoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ calling the decoder method
            x: tensor (batch, target_seq_len, dm) decoder input
            encoder_output:tensor(batch, input_seq_len, dm)
              output of the encoder
            training: bool for if in training
            look_ahead_mask: mask applied to first mha layaer
            padding_mask: mask for second mha layer
            Returns: tensor (batch, target_seq_len, dm) decoder output
        """
        seq_len = x.shape[1]
        attn_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training, look_ahead_mask,
                               padding_mask)
        return x
