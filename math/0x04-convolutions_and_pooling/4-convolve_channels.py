#!/usr/bin/env python3
""" colvolution on images with channels """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """convolve on img with channels
       images: np.ndarray (m, h, w, c)
            m: number of images, h: height in pixels, w: width in pixels
            c: number of channels
        kernel: np.ndarray (kh, kw, c) kernel for convolution
            kh: height of kernel, kw: width of kernel
        padding: tuple of (ph, pw) padding height and width
            or 'same' or 'valid'
        stride: tuple of (sh, sw) of the stride height and width
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride[0], stride[1]
    if type(padding) is tuple:
        ph, pw = padding[0], padding[1]
    elif padding == 'valid':
        ph, pw = 0, 0
        conved = np.zeros((m, h - kh + 1, w - kw + 1))
    else:
        ph = int((((h - 1) * sh) + kh - h) / 2 + 1)
        pw = int((((w - 1) * sw) + kw - w) / 2 + 1)
        # ph, pw = int((kh) / 2), int((kw) / 2)
    conh = int((h + 2 * ph - kh) / sh + 1)
    conw = int((w + 2 * pw - kw) / sw + 1)
    conved = np.zeros((m, conh, conw))
    padimg = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant',
                    constant_values=0)
    for i in range(0, conh):
        for j in range(0, conw):
            subs = padimg[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            conved[:, i, j] = np.sum((kernel[None, :, :, :] * subs),
                                     axis=(1, 2, 3))
    print(conved.shape)
    print(conved.shape[0])
    print(conved[0][0].shape)
    print(conved[0][0])
    print(images[0][0])
    return conved
