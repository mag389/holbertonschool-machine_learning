#!/usr/bin/env python3
""" bleu score file for ngrams """
import numpy as np


def ngram_bleu(references, sentence, n):
    """ calculates unigram BLEU score for a sentence
        references: list of reference translations
          each a list of words in the translation
        sentence: list contianing the model proposed sentence
        returns: the unigram bleu score
    """
    sen_len = len(sentence)
    ref_len = [len(ref) for ref in references]
    n_sent = ngramify(sentence, n)
    n_ref = [ngramify(ref, n) for ref in references]

    word_dict = {}
    for word in n_sent:
        max_count = 0
        if str(word) not in word_dict.keys():
            word_dict[str(word)] = 0
        for reference in n_ref:
            counts = reference.count(word)
            if counts > max_count:
                max_count = counts
            word_dict[str(word)] = min(max_count,
                                       max(counts, word_dict[str(word)]))
    closest = np.argmin(np.abs(np.array(ref_len) - sen_len))
    closest = references[closest]
    clo_len = len(closest)
    if clo_len < sen_len:
        bp = 1
    else:
        bp = np.exp(1 - clo_len / sen_len)
    return bp * sum(word_dict.values()) / len(n_sent)


def ngramify(sentence, n):
    """ creates ngram list from a sentence """
    listy = []
    for i in range(len(sentence) - n + 1):
        listy.append(sentence[i:i+n])
    return listy
