#!/usr/bin/env python3
""" sdp attention now """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ calculates scaled dot product attantion
        Q: tensor(..., seq_len_q, dk) of query matrix
        K: tensor(..., seq_len_v, dk) of key matrix
        V: tensor(..., seq_len_v, dv) of value matrix
        mask: tensor tht can be broadcast into (..., seq_len_q, seq_len_v)
          optional mask or None
        Returns:outputs, weights
          outputs: tensor(..., seq_len_q, dv) sdp attention
          weights: tensor(..., seq_len_q, dv) attention weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # then scale
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attn_logits = matmul_qk / (dk ** .5)
    # optional mask
    if mask is not None:
        scaled_attn_logits += (mask * -1e9)
    # softmax
    weights = tf.nn.softmax(scaled_attn_logits, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
