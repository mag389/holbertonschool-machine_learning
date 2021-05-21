#!/usr/bin/env python3
""" rnn decoder class file """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ rnn decoder class """
    def __init__(self, vocab, embedding, units, batch):
        """ initialization method for RNNdecoder object
            vocab: int: size of output vocabulary
            embedding:int: dimensionality of embedding vector
            units: int: number of hidden units in RNN cell
            batch: int: batch size
        """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """ call method for the decoder
            x: tensor (batch, 1) of previous word in target sequence
              as index of target vocab
            s_prev: tensor (batch, units) previous ecoder hidden state
            hidden_states: tensor (batch, niput_seq_len, units) encoder output
            Returns: y, s:
              y: tensor (batch, vocab) output as one hot vecor in target voacb
              s: tensor (batch, units) new decoder hidden state
        """
        sa = SelfAttention(self.units)
        context, weights = sa(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.F(output)
        return x, state
