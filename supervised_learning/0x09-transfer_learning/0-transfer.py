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
    # print(x_train.shape)

    # data augmentation
    datagener = False
    if datagener is True:
        datagen = K.preprocessing.image.ImageDataGenerator(
            # rescale made results significantly worse
            # rescale=1./255,
            zoom_range=0.3, rotation_range=50,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
            horizontal_flip=True, fill_mode='nearest'
        )
        datagen.fit(x_train)

    # create the model to transfer from
    classes = 10
    model = dense121(
        weights="imagenet",
        include_top=False,
        pooling='max',
        input_shape=(32, 32, 3))

    # model.summary()
    # K.utils.plot_model(model, to_file="model.png")

    # freeze model
    # model.trainable = False

    # add new layers
    input = K.Input(shape=(32, 32, 3))
    lambtha = K.layers.Lambda(lambda X: K.backend.resize_images(X, 7, 7,
                              data_format="channels_last")#,
                              # interpolation='bilinear')
                              )(input)
    model = dense121(
        weights="imagenet",
        include_top=False,
        pooling='max',
        input_shape=(224, 224, 3),
        input_tensor=lambtha
    )
    newl = K.layers.Dense(512, activation='relu')(model.output)
    newl = K.layers.Dropout(0.3)(newl)
    newl = K.layers.Dense(128, activation='relu')(newl)
    newl = K.layers.Dropout(0.3)(newl)
    newl = K.layers.Dense(10, activation='softmax')(newl)

    # new_model = K.Model(inputs=model.inputs, outputs=newl)
    # frz_model = K.Model(inputs=model.inputs, outputs=newl)
    frozen = False
    new_model = K.Model(inputs=input, outputs=newl)
    frz_model = K.Model(inputs=input, outputs=newl)

    # new_model.summary()

    # lacks accuracy so adding checkpoints
    calls = []
    calls.append(K.callbacks.ModelCheckpoint("cifar10.h5",
                 save_best_only=True))
    es = K.callbacks.EarlyStopping(monitor='val_loss', patience=7)
    calls.append(es)
    learn = K.callbacks.ReduceLROnPlateau(verbose=1, patience=3)
    calls.append(learn)

    # compile the model
    new_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc'])
    model.trainable = False
    frz_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['acc'])

    # train
    if datagener is True:
        new_model.fit(datagen.flow(x_train, y_train, batch_size=256),
                      # x_train, y_train,
                      validation_data=(x_test, y_test),
                      # use batch size when not using datagen
                      # batch_size=256,
                      # steps_per_epoch=1,
                      epochs=30,
                      verbose=1,
                      callbacks=calls)
    elif frozen is True:
        frz_model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      batch_size=256,
                      epochs=3,
                      verbose=1,
                      callbacks=calls,
                      # shuffle is unused in datagen
                      shuffle=False)
        new_model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      batch_size=256,
                      epochs=1,
                      verbose=1,
                      callbacks=calls,
                      # shuffle is unused in datagen
                      shuffle=False)
        frz_model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      batch_size=256,
                      epochs=3,
                      verbose=1,
                      callbacks=calls,
                      # shuffle is unused in datagen
                      shuffle=False)
        new_model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      batch_size=256,
                      epochs=1,
                      verbose=1,
                      callbacks=calls,
                      # shuffle is unused in datagen
                      shuffle=False)
        frz_model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      batch_size=256,
                      epochs=8,
                      verbose=1,
                      callbacks=calls,
                      # shuffle is unused in datagen
                      shuffle=False)
    else:
        frz_model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      batch_size=256,
                      epochs=10,
                      verbose=1,
                      callbacks=calls,
                      shuffle=False)
    new_model.save("cifar10.h5")
