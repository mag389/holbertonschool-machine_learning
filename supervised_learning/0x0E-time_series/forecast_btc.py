#!/usr/bin/env python3
""" use keras to predict btc price based on previous 24h data """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# first we need to use functions to get data into a tf dataset
def split_window(features):
    """ healper function for mapping data as inputs and labels
        features: the timeseries dataset as array
        Returns: inputs, labels
          inputs: list of the time series for training
          labels: list of the corresponding labels
    """
    input_w = 24 * 60
    label_w = 1
    label_start = 24 * 60 + 59
    input_slice = slice(0, input_w)
    label_slice = slice(label_start, None)

    inputs = features[:, input_slice, :]
    labels = features[:, label_slice, :]
    labels = labels[:, :, -1:]

    inputs.set_shape([None, input_w, None])
    labels.set_shape([None, label_w, None])
    return inputs, labels


def keras_data(dataframe):
    """ change pandas dataframe into keras detaset
        dataframe: a pandas dataframe, preprocesses, but not normalized
        Returns: the tf.dataset.Dataset
    """
    # if timestamp column remains remove
    # datafram = dataframe.drop(labels=['Timestamp'], axis='columns')

    # normalize dataframe data in place
    dataframe = (dataframe - dataframe.mean()) / dataframe.std()

    # conver to numpy
    data_arr = np.array(dataframe, dtype=np.float32)

    # make tf timeseries from the np array
    # this particular set of data has an offset of 8 minutes
    sequence_length = 60 * 25
    # this is only keras >2
    ids = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_arr,
        targets=None,
        sequence_length=sequence_length,
        sequence_stride=60,
        batch_size=64,
        start_index=8)
    ids = ids.map(split_window)
    return ids


def make_model():
    """ create the keras model for training later
        useful to have as separate function to easily change model later
    """
    keras = tf.keras
    model = keras.models.Sequential()
    # potential extra lstm layer here
    model.add(keras.layers.LSTM(64, activation='relu', return_sequences=False))
    model.add(keras.layers.Dense(1))
    # es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
    #                                       mode='min')
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[tf.metrics.MeanAbsoluteError()])
    return model


if __name__ == "__main__":
    dfnoncor = pd.read_csv('processed_data.csv')
    n = len(dfnoncor)
    train_split = int(n * 0.8)

    # create separate training and validation training sets
    train_df = dfnoncor[:train_split]
    valid_df = dfnoncor[train_split:]

    # create tf datasets from them
    train_ds = keras_data(train_df)
    valid_ds = keras_data(valid_df)

    # create model
    dsmodel = make_model()

    # create earlys tsopping criteria
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                          mode='min')

    # train the model
    hist = dsmodel.fit(x=train_ds,
                       validation_data=valid_ds,
                       callbacks=[es],
                       epochs=20)
