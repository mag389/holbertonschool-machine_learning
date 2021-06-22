#!/usr/bin/env python3
# import tensorflow as tf
import tensorflow_hub as hub
# import transformers as tfs
# from transformers import BertTokenizer, AutoModelForQuestionAnswering
import numpy as np
import glob


def semantic_search(corpus_path, sentence):
    """ searches corpus of documents with a sentence
        returns: refence text most similar to sentence
    """
    use = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    embed = hub.load(use)
    embeddings = [sentence]

    for name in glob.glob(corpus_path + "/*"):
        with open(name) as f:
            embeddings += [f.read()]

    nums = embed(embeddings)
    corr = np.inner(nums, nums)
    am = np.argmax(corr[0, 1:])
    return embeddings[am + 1]
