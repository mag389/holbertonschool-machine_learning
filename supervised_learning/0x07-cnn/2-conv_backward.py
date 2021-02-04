#!/usr/bin/env python3
""" back prop over a convolution layer """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ back propagate over a convolution layer
        dZ: np.ndarray (m, h_new, w_new, c_new) partial deivs with respect
        to unactivted output of nn layer
            m: number of example
            h_new: height of the output
            w_new: width of the output
            c_new: nunmber of channels
        A_prev: np.ndarray (m, h_prev, w_prev, c_prev) output of prev layer
            h_prev, w_prev, c_prev: hieght width and number of channels of
            prev layer (respectively)
        W: np.ndarray(kh, kw, c_prev, c_new) kernels
            kh, kw: filter height, filter width
        b: np.ndarray (1, 1, 1, c_new) of biases
        padding: string of same of valid indicating padding
        stride: tuple (sh, sw) of stride for height and with
        Returns: new partial derivs (dA_prev), kernels(dW) and biases(db)
    """
    m, hn, wn, cn = dZ.shape
    _, h_p, w_p, cp = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    else:
        # ph = int(np.ceil((sh * (h_p - 1) - h_p + kh) / 2))
        # pw = int(np.ceil((sw * (w_p - 1) - w_p + kw) / 2))
        # this second mthod was mine originally that didn't work
        # ph = int((((h_p - 1) * sh) + kh - h_p) / 2)
        # pw = int((((w_p - 1) * sw) + kw - w_p) / 2)
        ph = int(np.ceil(((sh * h_p) - sh + kh - h_p) / 2))
        pw = int(np.ceil(((sw * w_p) - sw + kw - w_p) / 2))

    conh = int((h_p + 2 * ph - kh) / sh + 1)
    conw = int((w_p + 2 * pw - kw) / sw + 1)
    conved = np.zeros((m, conh, conw, cn))
    padimg = np.pad(A_prev,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant',
                    constant_values=0)
    dx = np.zeros(padimg.shape)
    dW = np.zeros(W.shape)
    # db = np.zeros(b.shape)

    for i in range(m):
        for j in range(hn):
            for k in range(wn):
                for l in range(cn):
                    dx[i, j * sh:j * sh + kh, k * sw:k * sw + kw,
                       :] += dZ[i, j, k, l] * W[:, :, :, l]
                    dW[:, :, :, l] += padimg[i,
                                             j * sh:j * sh + kh,
                                             k * sw:k * sw + kw,
                                             :] * dZ[i, j, k, l]
                    # db[:, :, :, l] += dZ[i, j, k, l]
    if padding == 'same':
        dx = dx[:, ph:dx.shape[1] - ph, pw:dx.shape[2] - pw, :]
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    return dx, dW, db
