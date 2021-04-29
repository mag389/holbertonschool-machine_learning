#!/usr/bin/env python3
""" deep rcurrent neural network funciton file """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ performs forward propagation for a deep RNN
        rnn_cells: list of RNNCell instances of length l usef for forward prop
          l: number of layers
        X: data to be used, np arr (t, m, i)
          t: maximum number of time steps
          m: batch size
          i: deminsionality of the data
        h_0: initial hidden state np arr (l, m, h)
          h: dimensionality of hidden state
        returns: H, Y
          H: np arr of all hidden states
          Y np arr of all outputs
    """
    la = len(rnn_cells)
    t, m, i = X.shape
    _, _, h = h_0.shape

    # initialize H and Y
    H = np.zeros((t + 1, la, m, h))
    last_cell = rnn_cells[-1]
    lc_o = last_cell.by.shape[1]
    Y = np.zeros((t, m, lc_o))  # 6, 8, 5 = time, batch, output shape
    H[0] = h_0
    for it in range(t):
        # print("time: ", it)
        for lt in range(la):
            # print("cell: ", lt)
            cell = rnn_cells[lt]
            if lt == 0:
                xh = X[it]
            else:
                # xh = y_mid
                xh = H[it + 1, lt - 1]
            # print("shape: ", xh.shape)
            h, y_mid = cell.forward(H[it, lt], xh)
            H[it + 1, lt] = h
        Y[it] = y_mid
    return H, Y
