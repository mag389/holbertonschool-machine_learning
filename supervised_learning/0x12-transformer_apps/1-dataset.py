#!/usr/bin/env python3
""" dataset class file with new method """
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
