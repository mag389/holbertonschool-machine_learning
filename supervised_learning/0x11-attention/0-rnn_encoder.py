#!/usr/bin/env python3
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ the rnn encoder class for and encode decode arch
        using RNN's
    """
    def __init__(self, vocab, embedding, units, batch):
        """ initialize rnnecoder object
             (therefore also super)
            vocab: int of size of input vocab
            embedding: int of dimensionality of embedding vector
            units: int of number of hidden units in RNN cell
            batch: int of batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        # returns fill seq and last hidden state
        # recurrent weights init with glorot_uniform(default)
        self.gru = tf.keras.layers.GRU(units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """ init hidden state for the RNN cell to a tesor of zeros
            Returns: a tensor (batch, units) of initialized hidden states
              tensor is type tf.Tensor
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """ call method of the rnn encoder
            x: tensor of shape (batch, input_seq_len) of input to encoder
              layer as word indices within the vocab
            initial: tensor of (batch, units) initial hidden state
            Returns: outputs, hidden
              outputs: tensor (batch, input_seq_len) of encoder outputs
              hidden: tensor (batch, units) of last hidden state of encoder
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=initial)
        return output, state
