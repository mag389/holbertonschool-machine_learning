#!/usr/bin/env python3
""" bidirectional RNN's """
import numpy as np


class BidirectionalCell():
    """ bidirectional cell """
    def __init__(self, i, h, o):
        """ bidirectional cell contructor
            i: dimensionality of data
            h: dimensionality of hidden states
            o: dimensinality of outputs
          fields: (weights and biases)
            Whf, bhf: hidden states in forward direction
            Whb, bhb: hidden states in backward direction
            Wy, by: outputs
        """
        self.Whf = np.random.randn(h + i, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(h + i, h)
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ calculates hidden state in forward direction for one time step
            h_prev: np arr (m, h) of previous hidden state
            x_t: np arr (m, i) of data input of cell
              m: batch size of data
            Returns: h_next: the next hidden state
        """
        m, _ = x_t.shape
        h = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(h @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """ calculates hidden state in backward direction for one time step
            h_next: np arr (m, h) of next hidden state
            x_t: np arr (m, i) of data input of cell
              m: batch size of data
            Returns: h_prev: the previous hidden state
        """
        h = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(h @ self.Whb + self.bhb)
        return h_prev
