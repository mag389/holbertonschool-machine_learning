#!/usr/bin/env python3
import gensim as gs
import numpy as np


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """ creates and trains a genism fastText model
        sentences: list of sentences to be trained on
        size: dimensionality of embedding layer
        min_count: min no occurences of a word for use in training
        window: max distance between current and predicted word in a sentence
        negative: size of negative sampling
        cbow: a bool to determine training type(true is SBOW, false is Skipgram
        iterations: number of iterations to rian over
        seed: seed for random number gen
        workers: number of worker threads to trian on
        Returns: the trained model
    """
    ft = gs.models.fasttext
    sg = 1 - 1 * cbow
    model = ft.FastText(sentences, sg=sg, min_count=min_count,
                        window=window, negative=negative, seed=seed,
                        workers=workers, size=size, iter=iterations)
    return model
