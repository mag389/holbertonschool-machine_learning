#!/usr/bin/env python3
""" crop the image in tf """
import tensorflow as tf


def crop_image(image, size):
    """ crops the given image to size """
    return tf.image.random_crop(image, size)
