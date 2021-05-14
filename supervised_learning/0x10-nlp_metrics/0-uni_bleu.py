#!/usr/bin/env python3
""" bleu score file """
import numpy as np


def uni_bleu(references, sentence):
    """ calculates unigram BLEU score for a sentence
        references: list of reference translations
          each a list of words in the translation
        sentence: list contianing the model proposed sentence
        returns: the unigram bleu score
    """
    sen_len = len(sentence)
    ref_len = [len(ref) for ref in references]
    bps = np.zeros(len(ref_len))

    word_dict = {}
    for word in sentence:
        max_count = 0
        if word not in word_dict.keys():
            word_dict[word] = 0
        for reference in references:
            counts = reference.count(word)
            if counts > max_count:
                max_count = counts
            word_dict[word] = min(max_count, max(counts, word_dict[word]))
    closest = np.argmin(np.abs(np.array(ref_len) - sen_len))
    closest = references[closest]
    clo_len = len(closest)
    if clo_len < sen_len:
        bp = 1
    else:
        bp = np.exp(1 - clo_len / sen_len)
    return bp * sum(word_dict.values()) / sen_len
