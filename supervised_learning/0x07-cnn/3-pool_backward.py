#!/usr/bin/env python3
""" back propagation on pool layer """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs back propagation over a pooling layer
        dA: np.ndarray(m, h_new, w_new, c_new) of partil derivs of pooling
          layer with respect to outputs
            m: number of examples
            h_new: hieght of output
            w_new: width of output
            c: number of channels
        A_prev: np.ndarray(m, h_prev, w_prev, c) output of rpev layer
            h_prev: height of prev layer
            w_prev: width of prev layer
        kernel_shape: tuple of kh, kw: size fo kernel
            kh, kw: kernel height, kernel width
        stride: tuple of strides (sh, sw)
            sh, sw: stride for height, stride for width
        mode: string of max or avg if how to do pooling
        returns partial derivs of prev layer: dA_prev
    """
    m, h_n, w_n, c = dA.shape
    _, h_p, w_p, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    conh = int((h_p - kh) / sh + 1)
    conw = int((w_p - kw) / sw + 1)
    coved = np.zeros((m, conh, conw, c))

    dx = np.zeros(A_prev.shape)

    for i in range(m):
        for j in range(h_n):
            for k in range(w_n):
                for l in range(c):
                    if mode == 'max':
                        tmp = A_prev[i,
                                     j * sh:j * sh + kh,
                                     k * sw:k * sw + kw,
                                     l]
                        maxes = (tmp == np.max(tmp))
                        dx[i,
                           j * sh:j * sh + kh,
                           k * sw:k * sw + kw,
                           l] += dA[i, j, k, l] * maxes
                    if mode == 'avg':
                        dx[i,
                           j * sh:j * sh + kh,
                           k * sw:k * sw + kw,
                           l] += dA[i, j, k, l] / (kh * kw)
    return dx
