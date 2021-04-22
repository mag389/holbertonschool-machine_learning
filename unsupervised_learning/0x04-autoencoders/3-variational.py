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
    for i in range(1, len(hidden_layers)):
        dims = hiddel_layers[i]
        h = keras.layers.Dense(dims, activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)
    """
    def sampling(args):
        """ sampling function for vae
            args: mean, logvar
        """
        z_mean, z_log_sigma = args
        K1 = keras.backend.shape(z_mean)[0]
        epsilon = keras.backend.random_normal(shape=(K1, latent_dims),
                                              mean=0, stddev=1)
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon
    """
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0,
            stddev=1
        )
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon
    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z])

    dinput = keras.Input(shape=(latent_dims,))
    dh = keras.layers.Dense(hidden_layers[-1], activation='relu')(dinput)
    for i in range(len(hidden_layers) - 2, -1, -1):
        dims = hidden_layers[i]
        dh = keras.layers.Dense(dims, activation='relu')(dh)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(dh)

    decoder = keras.Model(dinput, decode)
    outputs = decoder(encoder(inputs)[-1])
    auto = keras.Model(inputs, outputs)

    def vae_loss(true, pred):
        """ a separate custom loss fucntion """
        rloss = keras.losses.binary_crossentropy(inputs, outputs)
        rloss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) \
            - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(rloss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
