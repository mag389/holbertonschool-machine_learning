#!/usr/bin/env python3
""" variational autoencoder """
import tensorflow.keras as keras


# def sampling(args):
#     """ sampling function for vae
#        args: mean, logvar
#        dim: latent dimensions
#    """
#    # z_mean, z_log_sigma, dim = args
#    z_mean, z_log_sigma = args
#    K = keras.backend.shape(z_mean)[0]
#    dim = keras.backend.int_shape(z_mean)[1]
#    epsilon = keras.backend.random_normal(shape=(K, dim),
#                                          mean=0, stddev=0.1)
#    return z_mean + keras.backend.exp(z_log_sigma) * epsilon


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
    latent = keras.layers.Dense(latent_dims, activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)

    def sampling(args):
        """ sampling function for vae
            args: mean, logvar
            dim: latent dimensions
        """
        z_mean, z_log_sigma = args
        K = keras.backend.shape(z_mean)[0]
        # dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(K, latent_dims),
                                              mean=0, stddev=0.1)
        return z_mean + keras.backend.exp(z_log_sigma) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    # z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma, latent_dims])

    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z])

    dinput = keras.Input(shape=(latent_dims,))
    dh = keras.layers.Dense(hidden_layers[-1], activation='relu')(dinput)
    for i in range(len(hidden_layers) - 2, -1, -1):
        dims = hidden_layers[i]
        dh = keras.layers.Dense(dims, activation='relu')(dh)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(dh)
    decoder = keras.Model(dinput, decode)
    decoded = decoder(encoder(inputs))
    """
    def kl_reconstruction_loss(true, pred):
        reconstruction_loss = keras.losses.binary_crossentropy(inputs,
                                                               decoded)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) -\
            keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)
    vae_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, vae_outputs)
    auto.compile(optimizer='adam', loss=kl_reconstruction_loss)
    return encoder, decoder, auto
    """
    # create a separate loss f'n
    def loss_f(true, pred):
        """ the custom loss fucntion """
        rloss = keras.losses.binary_crossentropy(inputs, decoded)
        rloss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(rloss + kl_loss)
    # create the custom loss function
    rloss = keras.losses.binary_crossentropy(inputs, decoded)
    rloss *= input_dims
    kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean)
    kl_loss -= keras.backend.exp(z_log_sigma)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(rloss + kl_loss)
    # add the loss function to the model
    auto = keras.Model(inputs, decoded)
    # auto.add_loss(vae_loss)
    auto.compile(optimizer='adam', loss=loss_f)
    return encoder, decoder, auto
