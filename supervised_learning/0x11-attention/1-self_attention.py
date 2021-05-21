#!/usr/bin/env python3
""" self attention class file(bhadanau) """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ class to calculate attention for machine translation """

    def __init__(self, units):
        """
        Class constructor
        parameters:
            units [int]:
                represents the number of hidden units in the alignment model
        sets the public instance attributes:
            W: a Dense layer with units number of units,
                to be applied to the previous decoder hidden state
            U: a Dense layer with units number of units,
                to be applied to the encoder hidden state
            V: a Dense layer with 1 units,
                to be applied to the tanh of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

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
            self.U(hidden_states)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, attention_weights
