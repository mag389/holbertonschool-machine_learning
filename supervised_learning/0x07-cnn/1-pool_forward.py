#!/usr/bin/env python3
""" forward propagates pooling layer """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ forward props over a pool layer
        A_prev: np.ndarray (m, h_prev, w_prev, c_prev) output of prev layer
            m: number of examples
            h_prev: height of prev layer
            w_prev: width of prev layer
            c_prev: number of channels of prev layer
        kernel_shape: tupe (kh, kw) size of kernel
            kh, kw = kernel hieght, kernel width
        stride: tuple (sh, sw) of stride values
            sh, sw = stride height, stride for width
        mode: string of 'max' or 'avg' to determine pool method
        Returns: output of ppool layer
    """
    m, hp, wp, cp = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    conh = int((hp - kh) / sh + 1)
    conw = int((wp - kw) / sw + 1)
    conved = np.zeros((m, conh, conw, cp))

    for i in range(conh):
        for j in range(conw):
            if mode == 'avg':
                temp = np.average(A_prev[:,
                                         i * sh:i * sh + kh,
                                         j * sw:j * sw + kw, :],
                                  axis=(1, 2))
            elif mode == 'max':
                temp = np.amax(A_prev[:,
                                      i * sh:i * sh + kh,
                                      j * sw:j * sw + kw, :],
                               axis=(1, 2))
            conved[:, i, j, :] = temp
    return conved
