#!/usr/bin/env python3
""" sparse autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ creates and autoencoder
        input_dims: integer containing dimensions of model
        hidden_layers: list of number of nodes for each hidden layer
          layers should be reversed for decoder
        latent_dims: integer of dimensions of latent space representation
        lambtha: regularization param for L1 regularizing encoded putput
        Returns: encoder, decover, auto
          encoder: encoder model
          decoder: decoder model
          auto: full autoencoder model
    """
    regularizers = keras.regularizers
    input_l = keras.Input(shape=(input_dims,))
    newlayer = keras.layers.Dense(hidden_layers[0], activation='relu')(input_l)
    for i in range(1, len(hidden_layers)):
        dims = hidden_layers[i]
        newlayer = keras.layers.Dense(dims, activation='relu')(newlayer)
    laten_l = keras.layers.Dense(latent_dims,
                                 activation='relu',
                                 activity_regularizer=regularizers.l1(lambtha)
                                 )(newlayer)
    encoder = keras.Model(input_l, laten_l)

    lat_i = keras.Input(shape=(latent_dims,))
    declayer = keras.layers.Dense(hidden_layers[-1], activation='relu')(lat_i)
    for i in range(len(hidden_layers) - 2, -1, -1):
        dims = hidden_layers[i]
        declayer = keras.layers.Dense(dims, activation='relu')(declayer)
    decode_l = keras.layers.Dense(input_dims, activation='sigmoid')(declayer)
    decoder = keras.Model(lat_i, decode_l)

    auto = keras.Model(input_l, decoder(encoder(input_l)))
    auto.compile('adam', 'binary_crossentropy')
    return encoder, decoder, auto
