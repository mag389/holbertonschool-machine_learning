#!/usr/bin/env python3
""" performs convolution on grayscale images with potential pad"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_padding(images, kernel, padding):
    """ performs convolution on images based on kernel
        images: np.ndarray (m, h, w)
            m: number of images, h: height in pixels, w: width in pixels
        kernel: np.ndarray (kh, kw) kernel for convolution
            kh: height of kernel, kw: width of kernel
        padding: tuple of (ph, pw) padding height and width
        Returns: np.ndarray of convolved image
    """
    imgshape = images.shape
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]
    conved = np.zeros((imgshape[0], h - kh + 1 + 2 * ph, w - kw + 1 + 2 * pw))
    # conved = np.zeros((imgshape[0], h + ph, w + pw))
    ipad = ph
    jpad = pw
    # print(conved.shape)
    # print(kernel.shape, images.shape)
    # print(kernel[None, :, :].shape)
    for i in range(0, h - kh):
        for j in range(0, w - kw):
            subs = images[:, i:i + kh, j:j + kw]
            ip = i + ipad
            jp = j + jpad
            conved[:, ip, jp] = np.sum((kernel[None, :, :] * subs),
                                       axis=(1, 2))

    return conved
