# Time series forcasting with recurrent NN's

This folder is for designing and implementing a RNN to predict a time series.

Due to hardware constraints much of this was completed using google Colab and
 tensorflow 2 as opposed to tensorflow 1 the usual version for this repository

additional resources:
I have attempted to save the links of helpful resources and solutions used here

When using the preprocessing function for time series data, this error may appear
https://stackoverflow.com/questions/50809257/returning-dataset-from-tf-data-dataset-map-causes-tensorslicedataset-object
 The input shape was incorrect and had to be changed in the arguments


This was my first time using pandas (and so therefore first time using pandas dataframes)
They proved to be extremely useful
https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pop.html

Tensorflow does have a time series tutorial, it can't be directly used here, but was very

useful as reference: https://www.tensorflow.org/tutorials/structured_data/time_series#4_create_tfdatadatasets

Some people have had issues with how timeseries_data_from_array() function works
https://github.com/tensorflow/tensorflow/issues/44592  There are some who prefer the deprecated version
This gives a very good description of how both work.

For preprocessing with missing values there are many approaches one may take.
https://www.kaggle.com/juejuewang/handle-missing-values-in-time-series-for-beginners
often referred to as imputation. I used linear interpolation, that has advantages but may
not be the best

As always machine learning mastery has great resources on creating LSTM based RNN's
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
This will be more useful when going even further than this project to stacked (Deep)
LSTM networks

The best resource to learn about imputation is this: https://drnesr.medium.com/filling-gaps-of-a-time-series-using-python-d4bfddd8c460
He goes in depth about the different methods and advantages of each. Even if you don't
know what it means give this a read.

here's the much talked about function: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/timeseries_dataset_from_array

here's a useful example of an RNN implementation: http://www.dinalherath.com/2019/rnn/
but he uses very different methods to create one. Not very applicable here, but useful.

Though we're doing timeseries' here, it's very similar to text
https://www.tensorflow.org/tutorials/text/text_classification_rnn#stack_two_or_more_lstm_layers

