#!/usr/bin/env python3
""" change image brightness """
import tensorflow as tf


def change_brightness(image, max_delta):
    """ randomly changes brightnes of img
        image: 3d tf tensor of img to change
        max_delta: max amount the image should be brightened
    """
    return tf.image.adjust_brightness(image, max_delta)
