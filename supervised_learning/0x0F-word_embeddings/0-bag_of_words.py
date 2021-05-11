#!/usr/bin/env python3
""" creates bag of words embedding matrix """
import gensim as gs
import numpy as np
import sklearn.feature_extraction as skfe


def bag_of_words(sentences, vocab=None):
    """ creates a bag of words embedding matrix
        sentences: list of sentences to analyze
        vocab: list of vocab words to use
          if None all words in sentences hsoud be used
        returns: embeddings, features
          embeddings: np arr (s, f) of embeddings
            s: number of sentences
            f: number of features analyzed
          features: list of features used for embedding
    """
    vectorizer = skfe.text.CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features
