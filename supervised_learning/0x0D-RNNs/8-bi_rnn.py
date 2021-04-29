#!/usr/bin/env python3
""" forward prop for entire Bi RNN function file """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ performs forward propagation for bidirectional RNN
        bi_cell: instance of BidirectionalCell for propagation
        X: data to be used, np arr (t, m, i)
          t: max time steps
          m: batch size
          i: dimensionality of the data
        h_0: initial hidden state in forward direction, np arr (m, h)
          h: dimensionality of hidden state
        h_t: initial hidden state for backward direction, np arr (m, h)
        Returns: H, Y
          H: np arr of concatenated hidden states
          Y: np arr of outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    od = bi_cell.by.shape[1]

    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))

    Hf[0] = bi_cell.forward(h_0, X[0])
    Hb[-1] = bi_cell.backward(h_t, X[-1])

    for it in range(1, t):
        Hf[it] = bi_cell.forward(Hf[it - 1], X[it])
        Hb[-1 - it] = bi_cell.backward(Hb[0 - it], X[-1 - it])

    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
