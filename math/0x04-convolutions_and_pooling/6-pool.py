#!/usr/bin/env python3
""" pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on images """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    conh = int((h - kh) / sh + 1)
    conw = int((w - kw) / sw + 1)
    conved = np.zeros((m, conh, conw, c))

    for i in range(conh):
        for j in range(conw):
            if mode == 'avg':
                temp = np.average(images[:,
                                         i * sh:i * sh + kh,
                                         j * sw:j * sw + kw, :],
                                  axis=(1, 2))
            elif mode == 'max':
                temp = np.amax(images[:,
                                      i * sh:i * sh + kh,
                                      j * sw:j * sw + kw, :],
                               axis=(1, 2))
            conved[:, i, j, :] = temp
    return conved
