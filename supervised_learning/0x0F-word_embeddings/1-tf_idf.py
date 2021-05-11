#!/usr/bin/env python3
""" tf idf embeddings """
import numpy as np
import sklearn.feature_extraction as skfe


def tf_idf(sentences, vocab=None):
    """ creates TF-IDF embedding
        sentences: list of sentences to analyze
        vocab: list of vocab words to use for analysis
          if none use allw ords from sentences
        returns: embeddings, features
          embeddings: np arr (s, f) of embedding
            s: number of sentences, f: number of features
          features: list of features
    """
    vectorizer = skfe.text.TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()
    return embeddings, features
