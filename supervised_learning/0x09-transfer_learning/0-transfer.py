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
    """ this does preprocessing, but for receiving streams of img data
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    """
    return x_r, y_r


if __name__ == "__main__":
    dense121 = K.applications.DenseNet121

    (x, y), (xv, yv) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x, y)
    x_test, y_test = preprocess_data(xv, yv)

    classes = 10
    model = dense121(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3),
        classes=10)

    model.summary()
    # K.utils.plot_model(model, to_file="model.png")
    model.save("cifar10.h5")
    # return model
    # model.save('cifar10.h5')



