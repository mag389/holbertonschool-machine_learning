#!/usr/bin/env python3
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ the dataset class for using with transformers """
    def __init__(self):
        """ class initializer
            uses ted_hrlr_translate/pt_to_en
            saves both english and portuguese tokenizers
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

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
