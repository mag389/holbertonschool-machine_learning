#!/usr/bin/env python3
""" function for creating the masks """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_padding_mask(seq):
    """ creates padding style mask """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    """ creates a look ahead mask """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inputs, target):
    """ creates the masks for training/validation
     inputs: tf.Tensor of shape (batch_size, seq_len_in)
       contains the input sentence
     target: tf.Tensor of shape (batch_size, seq_len_out)
       contains the target sentence
     Returns: encoder_mask, combined_mask, decoder_mask
         encoder_mask: tf.Tensor padding mask shape 
           (batch_size, 1, 1, seq_len_in) for encoder
         combined_mask: tf.Tensor of shape
           (batch_size, 1, seq_len_out, seq_len_out) used in the
           1st attention block in the decoder to pad and mask future tokens
           in the input received by the decoder.It takes the maximum between
           a lookaheadmask and the decoder target padding mask.
         decoder_mask:tf.Tensor pading mask (batch_size, 1, 1, seq_len_in)
           used in 2nd attn block in decoder
    """
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    look_ahead = create_look_ahead_mask(tf.shape(target)[1])
    dec_target = create_padding_mask(target)
    combined_mask = tf.maximum(look_ahead, dec_target)

    return encoder_mask, combined_mask, decoder_mask
