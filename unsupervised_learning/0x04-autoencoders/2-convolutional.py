#!/usr/bin/env python3
""" convolutional autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ creates a convolutional autoencoder
        input_dims: tuple of ints of dimensions of model input
        filters: list of number of filters for each conv layer in encoder
          reversed for decoder
        latent_dims: tuple of intsof dims of latent space representation
        encoder conv with kernel (3, 3) relu followed by 2x2 max pool
        decoder conv uses 3x3 filte same padding relu followed by 2x2 upsample
          second to last conv layer uses valid padding
          last conv has filters of input dims with sigmod and no upsample
        Returns: encoder, decover, auto
          encoder: encoder model
          decoder: decoder model
          auto: full autoencoder model (adam opt with binary crossentropy loss)
    """
    input_l = keras.Input(shape=(input_dims))

    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu',
                            padding='same')(input_l)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    for i in range(1, len(filters)):
        x = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(input_l, x)

    lat_i = keras.Input(shape=latent_dims)
    y = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu',
                            padding='same')(lat_i)
    y = keras.layers.UpSampling2D((2, 2))(y)
    for i in range(len(filters) - 2, 0, -1):
        y = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                padding='same')(y)
        y = keras.layers.UpSampling2D((2, 2))(y)

    # second to last layer uses valid padding
    y = keras.layers.Conv2D(filters[0], (3, 3), activation='relu',
                            padding='valid')(y)
    y = keras.layers.UpSampling2D((2, 2))(y)
    # last layer has input dims filters with sigmoid function
    y = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                            padding='same')(y)

    decoder = keras.Model(lat_i, y)

    auto = keras.Model(input_l, decoder(encoder(input_l)))
    auto.compile('adam', 'binary_crossentropy')
    return encoder, decoder, auto
