#!/usr/bin/env python3
""" Rnn forward prop function """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ perform forward propagation on the rnn cell
        rnn_cell: instance of RNNCell
        X: data to be used np arr (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state, np arr (m, h)
            h: dimensionality of hidden state
        returns: H, Y
            H: np arr of all hidden states
            Y: np arr of all outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.by.shape[1]))
    H[0] = h_0
    h = h_0
    for j in range(t):
        hi, y = rnn_cell.forward(H[j], X[j])
        # print("hi and y")
        # print(hi.shape)
        # print(y.shape)
        H[j + 1] = hi
        Y[j][:] = y
    # print(h_0)
    return H, Y
