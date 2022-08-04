# -*- coding: utf-8 -*-
"""
Created on Fri May  6 13:25:16 2022

@author: AzamaoZ
"""

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GlobalAveragePooling1D
from tensorflow.keras.constraints import max_norm
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Activation, SpatialDropout1D, GlobalMaxPooling1D, TimeDistributed
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam


# data1 = pd.read_csv('C:/Users/AzamaoZ/Desktop/S002R01_data.csv').T
# data2 = data1.to_numpy()
# data3 = np.expand_dims(data2, axis=0)
# data4 = data3[:,:13000,:]
# data1.shape;
# labels = [1]



data1 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S001R01_data.csv').T
data2 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S001R02_data.csv').T
data3 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S002R01_data.csv').T
data4 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S002R02_data.csv').T
data5 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S003R01_data.csv').T
data6 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S003R02_data.csv').T
data7 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S004R01_data.csv').T
data8 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S004R02_data.csv').T
data9 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S005R01_data.csv').T
data10 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S005R02_data.csv').T
data11 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S006R01_data.csv').T
data12 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S006R02_data.csv').T
data13 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S007R01_data.csv').T
data14 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S007R02_data.csv').T
data15 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S008R01_data.csv').T
data16 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S008R02_data.csv').T
data17 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S009R01_data.csv').T
data18 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S009R02_data.csv').T
data19 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S010R01_data.csv').T
data20 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S010R02_data.csv').T


# data1 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S001R01_data.csv')
# data2 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S001R02_data.csv')
# data3 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S002R01_data.csv')
# data4 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S002R02_data.csv')
# data5 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S003R01_data.csv')
# data6 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S003R02_data.csv')
# data7 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S004R01_data.csv')
# data8 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S004R02_data.csv')
# data9 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S005R01_data.csv')
# data10 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S005R02_data.csv')
# data11 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S006R01_data.csv')
# data12 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S006R02_data.csv')
# data13 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S007R01_data.csv')
# data14 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S007R02_data.csv')
# data15 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S008R01_data.csv')
# data16 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S008R02_data.csv')
# data17 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S009R01_data.csv')
# data18 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S009R02_data.csv')
# data19 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S010R01_data.csv')
# data20 = pd.read_csv('C:/Users/AzamaoZ/Desktop/WIFO/Subjects_EYE_open_closed/csv/S010R02_data.csv')

data1 = data1.to_numpy()
data2 = data2.to_numpy()
data3 = data3.to_numpy()
data4 = data4.to_numpy()
data5 = data5.to_numpy()
data6 = data6.to_numpy()
data7 = data7.to_numpy()
data8 = data8.to_numpy()
data9 = data9.to_numpy()
data10 = data10.to_numpy()
data11 = data11.to_numpy()
data12 = data12.to_numpy()
data13 = data13.to_numpy()
data14 = data14.to_numpy()
data15 = data15.to_numpy()
data16 = data16.to_numpy()
data17 = data17.to_numpy()
data18 = data18.to_numpy()
data19 = data19.to_numpy()
data20 = data20.to_numpy()

data1 = np.expand_dims(data1, axis=0)
data2 = np.expand_dims(data2, axis=0)
data3 = np.expand_dims(data3, axis=0)
data4 = np.expand_dims(data4, axis=0)
data5 = np.expand_dims(data5, axis=0)
data6 = np.expand_dims(data6, axis=0)
data7 = np.expand_dims(data7, axis=0)
data8 = np.expand_dims(data8, axis=0)
data9 = np.expand_dims(data9, axis=0)
data10 = np.expand_dims(data10, axis=0)
data11 = np.expand_dims(data11, axis=0)
data12 = np.expand_dims(data12, axis=0)
data13 = np.expand_dims(data13, axis=0)
data14 = np.expand_dims(data14, axis=0)
data15 = np.expand_dims(data15, axis=0)
data16 = np.expand_dims(data16, axis=0)
data17 = np.expand_dims(data17, axis=0)
data18 = np.expand_dims(data18, axis=0)
data19 = np.expand_dims(data19, axis=0)
data20 = np.expand_dims(data20, axis=0)




data_all = np.concatenate((data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12,
                         data13, data14, data15, data16, data17, data18, data19, data20),axis=0)

labels = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
labels = np.array(labels)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_all, labels, test_size = 0.2)

model = Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(800, activation = "relu"))
model.add(tf.keras.layers.Dense(500, activation = "relu"))
model.add(tf.keras.layers.Dense(500, activation = "relu"))
model.add(tf.keras.layers.Dense(500, activation = "relu"))
model.add(tf.keras.layers.Dense(10, activation = "relu"))
model.add(tf.keras.layers.Dense(2, activation = "softmax"))

optimizer = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# model_history = model.fit(data_all, labels, batch_size = 16, validation_split = 0.2, epochs = 50)
model_history = model.fit(x_train, y_train, batch_size = 16, validation_data = (x_test, y_test), epochs = 50)

model.summary()

data_all.shape







model_cnn = Sequential()

model_cnn.compile(optimizer= 'adam', loss= None)

model_cnn.add(tf.keras.layers.Conv1D(filters = 16, input_shape = (65, 9760), kernel_size = 16, activation = "relu"))
model_cnn.add(tf.keras.layers.BatchNormalization())
model_cnn.add(tf.keras.layers.MaxPooling1D(pool_size = 2))
model_cnn.add(tf.keras.layers.Dropout(0.2))

model_cnn.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 10, activation = "relu"))
model_cnn.add(tf.keras.layers.BatchNormalization())
model_cnn.add(tf.keras.layers.MaxPooling1D(pool_size = 2))
model_cnn.add(tf.keras.layers.Dropout(0.2))

model_cnn.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 4, activation = "relu"))
model_cnn.add(tf.keras.layers.BatchNormalization())
model_cnn.add(tf.keras.layers.Dropout(0.2))

model_cnn.add(tf.keras.layers.Flatten())
model_cnn.add(tf.keras.layers.Dense(10, activation = "relu"))
model_cnn.add(tf.keras.layers.BatchNormalization())
model_cnn.add(tf.keras.layers.Dropout(0.2))
model_cnn.add(tf.keras.layers.Dense(2, activation = "softmax"))





callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights = True)
optimizer = keras.optimizers.Adam(learning_rate = 1e-4)
model_cnn.compile(optimizer = 'adam', loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

keras.utils.plot_model(model_cnn, show_shapes = True);

model_cnn_history = model_cnn.fit(x_train, y_train, batch_size = 4, validation_data = (x_test, y_test), epochs = 100, callbacks = [callback])

keras.utils.plot_model(model_cnn, show_shapes = True)

test_cnn = model_cnn.evaluate(x_test, y_test)
model_cnn.summary()

# test = model.evaluate(x_test, y_test)
# test_cnn = model_cnn.evaluate(x_test, y_test)

# model_cnn_history = model_cnn.fit(data_all, labels, batch_size = 16, validation_split = 0.2, epochs = 50)