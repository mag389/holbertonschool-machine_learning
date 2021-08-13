#!/usr/bin/env python3
""" shear the image """
import tensorflow as tf


def shear_image(image, intensity):
    """ randomly shears an image
        image: 3d tensor of img to shear
        intensity: intensity with which to shear
        Returns: the sheared img
    """
    return tf.keras.preprocessing.image.random_shear(
        image.numpy(),
        intensity,
        row_axis=0,
        col_axis=1,
        channel_axis=2)
