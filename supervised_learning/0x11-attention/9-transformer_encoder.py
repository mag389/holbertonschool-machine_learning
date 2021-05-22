#!/usr/bin/env python3
""" tranformer encoder class file """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """ the encoder class for tranformer """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """ encoder initializer
            N: number of blocks in encoder
            dm: dimensionality of model
            h: number of heads
            hidden: number of units in FC layer
            input_vocab: size of input vocab
            max_seq_len - the maximum sequence length possible
            drop_rate - the dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for _ in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """ call the Encoder model object
            x: tensor of input data (batch, inpuy_seq_len, dm)
            training: bool of if it's training
            mask: for mha
            returns: tensor (batch, input_seq_len, dm) of encoder ouput
        """
        # seq_len = tf.shape(x)[1]
        seq_len = x.shape[1]
        # add position and embedding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)
        return x
