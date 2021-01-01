#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:19:46 2020

@author: katleyamedrano
"""

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.figure(figsize=(5,5))
for k in range(12):
    plt.subplot(3,4,k+1)
    plt.imshow(X_train[k], cmap="Greys")
    plt.axis("off")

plt.tight_layout()
plt.show()

#flatten two dimensional image to one-dimensional image
X_train = X_train.reshape(60000, 784).astype("float32")
X_test = X_test.reshape(10000, 784).astype("float32")

#normalize data #convert pixels to floats
X_train /= 255
X_test /= 255

#convert integer labels to one-hot #output  probabilities 
n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

#Keras code to architect shallow neural network
model = Sequential()   #simplest type of neural network model 
model.add(Dense(64, activation = "sigmoid", input_shape=(784,)))  #specify attributes of hidden layer
model.add(Dense(10, activation = "softmax"))  #specify output layer

model.summary()

(64*784)

(64*784)+64

(10*64)+10

#### Configure model

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

#train shallow neural network
model.fit(X_train, y_train, 
          batch_size=128, epochs=200, 
          verbose=1, validation_data=(X_test, y_test))



model.evaluate(X_test, y_test)
