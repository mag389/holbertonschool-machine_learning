#!/usr/bin/env python3
""" change image hue """
import tensorflow as tf


def change_hue(image, delta):
    """ change the image hue
        image: 3d image tensor
        delta: amount hue hsould change
    """
    return tf.image.adjust_hue(image, delta)
