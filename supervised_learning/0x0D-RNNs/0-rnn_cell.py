#!/usr/bin/env python3
""" RNN class file """
import numpy as np


class RNNCell():
    """ the RNN cell class """
    def __init__(self, i, h, o):
        """ the RNNCell constructor
            i: dimensionality of the data
            h: dimensions of hidden state
            o: dimensions of output
        """
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward propagate the RNNCell
            h_prev: np arr (m, h) of previous hidden state
            x_t: np arr (m, i) of data input for the cell
              m: batch size for the data
            softmax output
            returns: h_next, y
              h_next: next hidden state
              y: output of the cell
        """
        m, _ = h_prev.shape
        # our input uses h and x together
        h = np.concatenate((h_prev, x_t), axis=1)
        # print("printing shape")
        # print(h.shape)
        # print(self.Wh.shape)
        h_next = np.tanh(h @ self.Wh + self.bh)
        output = h_next @ self.Wy + self.by
        # softmax the output to get y
        y = np.exp(output - np.max(output))
        y = y / y.sum(axis=1)[:, np.newaxis]
        return h_next, y
