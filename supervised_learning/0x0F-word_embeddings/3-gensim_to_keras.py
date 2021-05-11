#!/usr/bin/env python3
""" converts gensim model to keras embedding layer """
import gensim as gs
import numpy as np
import tensorflow.keras as k


def gensim_to_keras(model):
    """ converts a gensim word2vec model to a keras Embedding layer
        model: trained gensim word2vec models
        Returns: the trainable keras Embedding
    """
    kmodel = model.wv.get_keras_embedding(train_embeddings=False)
    return kmodel
