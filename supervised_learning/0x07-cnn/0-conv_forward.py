#!/usr/bin/env python3
""" forward propagation of a convolution layer """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ forward props a convolution layer os a nn
        A_prev: np.ndarray(m, h_prev, w_prev, c_prev) output of prev layer
            m: number of examples
            h_prev: prev layer height
            w_prev: prev layer width
            c_prev: prev layer channels
        W: np.ndarray (kh, kw, c_prev, c_new) kernels for convolution
            kh: filter height
            kw: filter width
            c_prev: prev layer channels
            c_new: number of channels in the output
        b: np.ndarray (1, 1, 1, c_new) biases for convolution
        activation: act function applied to convolution
        padding: a string same or valid indication conv padding used
        stride: tuple (sh, sw) of strides
            sw, sw: stride for height and width respectively
        Returns: output of that conv layer
    """
    m, h_p, w_p, c_p = A_prev.shape
    kh, kw, c_p, c_new = W.shape
    sh, sw = stride[0], stride[1]
    if padding == 'valid':
        ph, pw = 0, 0
        # conved = np.zeros((m, h_p - kh + 1, w_p - kw + 1))
    else:
        ph = int((((h_p - 1) * sh) + kh - h_p) / 2)
        pw = int((((w_p - 1) * sw) + kw - w_p) / 2)

    conh = int((h_p + 2 * ph - kh) / sh + 1)
    conw = int((w_p + 2 * pw - kw) / sw + 1)
    conved = np.zeros((m, conh, conw, c_new))

    padimg = np.pad(A_prev,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant',
                    constant_values=0)
    for i in range(0, conh):
        for j in range(0, conw):
            # print("{}{}".format(i, j))
            for k in range(0, c_new):
                subs = padimg[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                conved[:, i, j, k] = np.sum((W[None, :, :, :, k] * subs),
                                            axis=(1, 2, 3))
                # conved[:, i, j, k] = activation(conved[:, i, j, k] +
                #                                 b[0, 0, 0, k])
    # return conved
    return activation(conved + b)
