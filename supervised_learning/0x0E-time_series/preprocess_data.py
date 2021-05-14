#!/usr/bin/env python3
""" preprocess the data from a csv file """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


# import the data from csv files as pandas
# bitdata = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
data = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

# then deal with NaN values via interpolation
# Here I use linear interpolation but it is not necessarily the best
# 'nearest' interpolation of forward padding can also be uses
# should be chosen bassed on results
df_all = data.interpolate(method='linear')
# can be useful to plot data to better understand at this point

# next we should remove the extraneous data features
# to determine extraneous features we can use a correlation matrix
"""
cormat = df_all.corr()
fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(cormat)
plt.xticks(range(len(cormat.columns)), cormat.columns);
plt.yticks(range(len(cormat.columns)), cormat.columns);
sns.heatmap(cormat, annot=True, cmap="YlGnBu")
plt.show()
"""
# with that block we can see which columns are exactly correlated with other
# then drop those columns
dfnoncor = df_all.drop(labels=['Open', 'High', 'Low', 'Close'], axis='columns')
# if desired plot again
# dfnoncor.Weighted_Price.plot(style=['b-.', 'ko', 'r.', 'rx-'],
#                              figsize=(20,10))

# finally we shold also drop the timestamp column because it's no longer needed
# dataframe = dataframe.drop(labels=['Timestamp'], axis='columns')
dfnoncor = dfnoncor.drop(labels=['Timestamp'], axis='columns')

# an optional step is to normalize the data, however that is performed later
# for this program. If normalizing as a pat of preprocessing here is when to
# normalize, however it may be advantageous to perform later in the process

# Finally resave to csv file (without index column)
dfnoncor.to_csv('processed_data.csv', index=False)
