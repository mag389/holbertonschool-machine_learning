#!/usr/bin/env python3
""" self attention class file """
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
        self.W = tf.layers.Dense(units)
        self.U = tf.layers.Dense(units)
        self.V = tf.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """ call funciton for self attention class
            s_prev: tensor(batch, units) previous decoder hidden state
            hidden_states: tensor(batch, input_seq_len, 1) of encoder outputs
            Returns: context, weights
              context:tensor(batch, units) of context vector of decoder
              weights: tensor(batch, input_seq_len, 1) of attention weights
        """
        query_with_time_axis = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(
            self.W(query_with_time_axis) +
            self.U(hidden_states)
        ))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, attention_weights
