#!/usr/bin/env python3
""" scr to train SNN to classify cifar 10, also functions used """
import tensorflow.keras as K


def preprocess_data(X, Y):
    """ preprocesses data for the model
        X: np.ndarray (m, 32, 32, 3) of cifar10 pictures
            m: number of examples/data points
        Y: np.ndarray (m,) cifar10 labels for X
        Returns X_p, Y_p: preprocessed X, Y
    """
    ImageDataGenerator = K.preprocessing.image.ImageDataGenerator
    y_r = K.utils.to_categorical(Y, 10)
    x_r = K.applications.densenet.preprocess_input(X)
    # data must be preprocessed differently for different CNN's
    # can replace the x_r function if using a different neural network arch
    return x_r, y_r


if __name__ == "__main__":
    dense121 = K.applications.densenet.DenseNet121

    (x, y), (xv, yv) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x, y)
    x_test, y_test = preprocess_data(xv, yv)
    print(x_train.shape)

    classes = 10
    model = dense121(
        weights="imagenet",
        include_top=False,
        pooling='max',
        input_shape=(32, 32, 3))

    # model.summary()
    # K.utils.plot_model(model, to_file="model.png")

    # freeze model
    model.trainable = False

    # add new layers
    newl = K.layers.Dense(256, activation='relu')(model.output)
    newl = K.layers.Dropout(0.3)(newl)
    newl = K.layers.Dense(10, activation='softmax')(newl)
    new_model = K.Model(inputs=model.inputs, outputs=newl)

    new_model.summary()

    # runs slowly so adding checkpoints
    calls = []
    calls.append(K.callbacks.ModelCheckpoint("cifar10.h5",
                 save_best_only=True))

    # compile the model
    new_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc'])
    # train
    new_model.fit(x_train, y_train, batch_size=256, epochs=1,
                  verbose=True,
                  callbacks=calls,
                  shuffle=False)
    new_model.save("cifar10.h5")
    # model.save('cifar10.h5')
