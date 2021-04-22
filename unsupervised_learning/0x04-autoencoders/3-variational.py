#!/usr/bin/env python3
""" variational autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a variational autoencoder
        input_dims: integer containing dims of model input
        hidden_layers: list containing number of nodes for each hidden layer
          reversed for decoder
        latent_dims: integer of dimensions of latent space representation
        returns: encoder, decoder, auto
          encoder: the encoder model, outputs latent representation, mean,
            and log variance, repectively
          decoder: decoder model
          auto: the full autoencoder model
    """
    inputs = keras.Input(shape=(input_dims,))
    h = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    for dims in hidden_layers[1:]:
        h = keras.layers.Dense(dims, activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)

    def sampling(args):
        """ samplig f'n for vae """
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0, stddev=1)
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z])

    dinputs = keras.Input(shape=(latent_dims,))
    x = keras.layers.Dense(hidden_layers[-1], activation='relu')(dinputs)
    for dims in hidden_layers[-2::-1]:
        x = keras.layers.Dense(dims, activation='relu')(x)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(dinputs, decode)
    outputs = decoder(encoder(inputs)[2])
    auto = keras.Model(inputs, outputs)

    def vae_loss(inputs, outputs):
        """ separate custom loss function """
        r_loss = keras.losses.binary_crossentropy(inputs, outputs)
        r_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) \
            - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(r_loss + kl_loss)
        return vae_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
