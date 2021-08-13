#!/usr/bin/env python3
""" flip image horizontally """
import tensorflow as tf


def flip_image(image):
    """ flips the image horizontally """
    return tf.image.flip_left_right(image)
