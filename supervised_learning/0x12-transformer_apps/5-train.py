#!/usr/bin/env python3
""" the train function file """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """ creates and trains a transformer model for machine translation
          of portuguese to English
        N: number of encoder and decoder blocks
        dm: dimensionality of the model
        h: number of heads
        hidden: number of hidden units in fc layers
        max_len: max tokens per sequence
        batch_size: batch size for training
        epochs: number of epochs

        adam opt (beta1=0.9, beta2=0.98, epsilon=1e-9)
        also special learning rat eequation
        sparse categorical crossentropy
        Returns: the trained model
    """
    ds = Dataset(batch_size, max_len)

    transformer = Transformer(
        N=N,
        dm=dm,
        h=h,
        hidden=hidden,
        input_vocab=ds.tokenizer_pt.vocab_size,
        target_vocab=ds.tokenizer_en.vocab_size,
        max_seq_input=max_len,
        max_seq_target=max_len
        )
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        """ custom loss function for transformer """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        """ single train step """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_p_mask, comb_mask, dec_p_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions = transformer(inp, tar_inp, True,
                                      enc_p_mask,
                                      comb_mask,
                                      dec_p_mask)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))
        train_loss(loss)

    for epoch in range(epochs):
        train_loss.reset_states()
        for (batch, (inp, tar)) in enumerate(ds.data_train):
            train_step(inp, tar)
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch}' +
                      ' Loss {train_loss.result():.4f}' +
                      ' Accuracy {train_accuracy.result():.4f}')
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}' +
              ' Accuracy {train_accuracy.result():.4f}')
    return transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ for decaying the lrate """
    def __init__(self, d_model, warmup_steps=4000):
        """ initializer """
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """ call method """
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
