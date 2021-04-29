#!/usr/bin/env python3
""" GRU class file """
import numpy as np


class GRUCell():
    """ the GRU cell class
        GRU: gated recurrent unit
    """
    def __init__(self, i, h, o):
        """ the GRUCell constructor
            i: dimensionality of the data
            h: dimensions of hidden state
            o: dimensions of output
            fields: (weights and biases)
            Wz, bz: for update gate
            Wr, br: for reset gate
            Wh, bh: for intermediate hidden state
            Wy, by: for output
        """
        self.Wz = np.random.randn(h + i, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(h + i, h)
        self.br = np.zeros((1, h))

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
        # calculate update gate
        in1 = h @ self.Wz + self.bz
        z = 1/(1 + np.exp(-1 *  in1))
        # and reset gate
        in2 = h @ self.Wr + self.br
        r = 1 / (1 + np.exp(-1 * in2))
        # then new hidden state
        coef = np.concatenate((r * h_prev, x_t), axis=1)
        h_temp = np.tanh((coef) @ self.Wh + self.bh)
        # finally new output
        print(z.shape, (1-z).shape)
        h_next = (1 - z) * h_prev + z * h_temp
        output = h_next @ self.Wy + self.by
        # softmax of the output
        y = np.exp(output - np.max(output))
        y = y / y.sum(axis=1)[:, np.newaxis]
        return h_next, y
        # code for vanilla RNN if you want to compare
        h_next = np.tanh(h @ self.Wh + self.bh)
        output = h_next @ self.Wy + self.by
        # softmax the output to get y
        y = np.exp(output - np.max(output))
        y = y / y.sum(axis=1)[:, np.newaxis]
        return h_next, y
