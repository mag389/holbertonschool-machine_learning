#!/usr/bin/env python3
""" third dataset class iteration for pipeline update of init """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ the dataset class for using with transformers """
    def __init__(self, batch_size, max_len):
        """ class initializer
            uses ted_hrlr_translate/pt_to_en
            saves both english and portuguese tokenizers
        """
        data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='train',
                               as_supervised=True)
        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='validation',
                               as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)
        # set datasets as tensors
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)
        # pipeline
        self.batch_size = batch_size
        self.max_len = max_len

        def make_train_batches(ds):
            """ make batches from the training set:
                filters to max_len, caches, shuffles, splits into padded batch
                prefetches data with experimental.AUTOTUNE
            """
            # need a variable to ise for shuffling must be larger than
            # data to shuffle in one batch, but can't be arbitrarily large
            # because of memory constraints
            data_size = sum(1 for i in self.data_train)
            return (
                ds
                .cache()
                .filter(lambda x, y: tf.size(x) <= max_len and
                        tf.size(y) <= max_len)
                .shuffle(data_size)
                .padded_batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))
        self.data_train = make_train_batches(self.data_train)

        def make_valid_batches(ds):
            """ make batches for validation
                only filters and splits into padded batches
            """
            return (
                ds
                .filter(lambda x: tf.size(x) <= max_len)
                .padded_batch(batch_size))
        self.data_valid = make_valid_batches(self.data_valid)

    def tokenize_dataset(self, data):
        """ creates sub-word tokenizers for the dataset
            data: tf.data.Dataset as tuple (pt, en)
              pt: tf.Tensor of portuguese sentence
              en: tf.Tensor of english sentence
            max vocab is 2^15
            Returns: tokenizer_pt, tokenizer_en
              tokenizer_pt: portuguese tokenizer
              tokenizer_en: english tokenizer
        """
        bfc = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_pt = bfc(
            (pt.numpy() for pt, _ in data),
            2**15)
        tokenizer_en = bfc(
            (en.numpy() for _, en in data),
            2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encodes a translation into tokens
            pt: tf.Tensor of portugueses sentence
            en: tf.Tensor of english sentence
            tokenized sentence should include start and end tokens
            start token indexed as vocab_size
            end token indexed as vocab_size + 1
            returns: pt_tokens, en_tokens
              pt_tokens: np arr of portuguese tokens
              en_tokens: np arr of English tokens
        """
        ptvs = self.tokenizer_pt.vocab_size
        envs = self.tokenizer_en.vocab_size
        pt_tok = self.tokenizer_pt.encode(pt.numpy())
        pt_tok = [ptvs] + pt_tok + [ptvs + 1]
        en_tok = self.tokenizer_en.encode(en.numpy())
        en_tok = [envs] + en_tok + [envs + 1]
        return pt_tok, en_tok

    def tf_encode(self, pt, en):
        """ Tf wrapper for encoding
            makes use of tf.py_function for encode
        """
        pt_tok, en_tok = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tok.set_shape([None])
        en_tok.set_shape([None])
        return pt_tok, en_tok
