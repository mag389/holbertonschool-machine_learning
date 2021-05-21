#!/usr/bin/env python3
""" self attention class file(bhadanau) """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ class to calculate attention for machine translation """

    def __init__(self, units):
        """ self attention initializer
            units: int number of hidden units in alignment model
            sets several public attributes:
          W: dense layer with units units (applt to prev decoder hidden state)
          U: dense layer with units (apply to encoder hidden states)
          V: Dense layer with 1 unit applied to tanh of the sum of outputs of
            W and U
            class inherits so call to super as well
        """
        super(SelfAttention, self).__init__()
        self.units = units
        if type(units) is not int:
            raise TypeError()
        self.W = tf.layers.Dense(units)
        self.U = tf.layers.Dense(units)
        self.V = tf.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Takes in previous decoder hidden state and outputs
            the context vector for decoder and attention weights
        parameters:
            s_prev [tensor of shape (batch, units)]:
                contains the previous decoder hidden state
            hidden_states [tensor of shape (batch, input_seq_len, units)]:
                contains the outputs of the encoder
        returns:
            context, weights:
                context [tensor of shape (batch, units)]:
                    contains the context vector for the decoder
                weights [tensor of shape (batch, input_seq_len, 1)]:
                    contains the attention weights
        """
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))
        weights = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
