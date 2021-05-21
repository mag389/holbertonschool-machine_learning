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
        self.W = tf.layers.Dense(units)
        self.U = tf.layers.Dense(units)
        self.V = tf.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Instance Call
        Arguments:
            s_prev {tf.Tensor} -- Is containing the previous decoder hidden
            state of shape (batch, units).
            hidden_states {tf.Tensor} -- Is Contatining the outputs of the
            of shape (batch, input_seq_len, units)
        Returns:
            tuple -- Contains a tf.Tensor contains context vector for the
            decoder of shape (batch, units), and tf.Tensor contains the
            attention weights of shape (batch, input_seq_len, 1).
        """
        s_prev_time = tf.expand_dims(s_prev, 1)
        score = self.V(
            tf.nn.tanh(
                self.W(s_prev_time) + self.U(hidden_states)
            )
        )
        weights = tf.nn.softmax(score, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, 
