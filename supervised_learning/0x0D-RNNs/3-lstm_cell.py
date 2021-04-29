#!/usr/bin/env python3
""" LSTM class file (long short term memory) """
import numpy as np


class LSTMCell():
    """ the LSTM cell class
        LSTM: long short term memory
        an alternate method from GRU's to tackle gradients
        (exploding and diminishing gradient problem)
    """
    def __init__(self, i, h, o):
        """ the LSTMCell constructor
            i: dimensionality of the data
            h: dimensions of hidden state
            o: dimensions of output
          fields: (weights and biases)
            Wf, bf: for forget gate
            Wu, bu: for update gate
            Wc, bc: for intermediate cell state
            Wo, bo: for output gate
            Wy, by: for output
        """
        self.Wf = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(h + i, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(h + i, h)
        self.bc = np.zeros((1, h))

        self.Wo = np.random.randn(h + i, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """ forward propagate the LSTMCell
            h_prev: np arr (m, h) of previous hidden state
            c_prev: np arr (m, h) of previous cell state
            x_t: np arr (m, i) of data input for the cell
              m: batch size for the data
            softmax output
            returns: h_next, c_next, y
              h_next: next hidden state
              y: output of the cell
        """
        m, _ = h_prev.shape
        # LSTM
        # concat new x and h_prev
        h = np.concatenate((h_prev, x_t), axis=1)
        # start with forget gate
        ft = h @ self.Wf + self.bf
        f = 1 / (1 + np.exp(-ft))
        # then update gate
        ug = h @ self.Wu + self.bu
        u = 1 / (1 + np.exp(-ug))
        # output gate
        og = h @ self.Wo + self.bo
        o = 1 / (1 + np.exp(-og))
        # intermediate cell state
        ci = h @ self.Wc + self.bc
        c_t = np.tanh(ci)
        # calculate new c (start by applying forget gate and update gate
        c_next = f * c_prev + u * c_t
        # calculate new h
        h_next = o * np.tanh(c_next)
        # and new output
        output = h_next @ self.Wy + self.by
        # and give softmax
        y = np.exp(output - np.max(output))
        y = y / y.sum(axis=1)[:, np.newaxis]
        return h_next, c_next, y
        # our input uses h and x together
        h = np.concatenate((h_prev, x_t), axis=1)
        # calculate update gate
        in1 = h @ self.Wz + self.bz
        z = 1 / (1 + np.exp(-1 * in1))
        # and reset gate
        in2 = h @ self.Wr + self.br
        r = 1 / (1 + np.exp(-1 * in2))
        # then new hidden state
        coef = np.concatenate((r * h_prev, x_t), axis=1)
        h_temp = np.tanh((coef) @ self.Wh + self.bh)
        # finally new output
        # print(z.shape, (1-z).shape)
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
