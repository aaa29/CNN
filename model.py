#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:10:36 2018

@author: amine
"""

import numpy as np

from sklearn.model_selection import train_test_split
from data_utils import get_sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils


#meta-data
#learning data
#model parameters
#
batch_size = 30
epochs = 20
#
nb_filters = 32
kernel_size = 2
#
nb_class = 3

X, Y = get_sequence(2000)
X = np.reshape(X, [X.shape[0], X.shape[1], 1])


Y = np_utils.to_categorical(Y, nb_class)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

model = Sequential()
model.add(Conv1D(filters = nb_filters, kernel_size = kernel_size, input_shape=[batch_size,X.shape[2]]))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_class))
model.add(Activation('softmax'))

model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x = X_train,
          y = Y_train,
          epochs = epochs,
          validation_data = [X_test, Y_test],
          batch_size = batch_size)