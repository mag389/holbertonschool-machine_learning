#!/usr/bin/env python3
""" save and load config """
import tensorflow.keras as K


def save_config(network, filename):
    """ saves model's config in JSON """
    with open(filename, "w") as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """ loads a model from file with configuration """
    with open(filename, "r") as f:
        model = K.models.model_from_json(f.read())
    return model
