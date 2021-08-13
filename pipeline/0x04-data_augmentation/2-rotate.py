#!/usr/bin/env python3
""" rotate the image """
import tensorflow as tf


def rotate_image(image):
    """ rotates an image 90 degrees ccw
        image: 3d tf.tensor of image
        Return: the rotated image
    """
    return tf.image.rot90(image)
