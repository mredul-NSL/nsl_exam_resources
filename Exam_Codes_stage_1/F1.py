# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:20:19 2021

@author: Mredul
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential

def format_input(i):
    return [1, i, i*i]


def preprocess():
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_train = np.array([5*i**2 + 7*i + 9 for i in values])
    x_train = np.array([format_input(i) for i in values])
    print(x_train.shape)
    print(y_train.shape)

def train_model():
    model = Sequential()
    model.add(Dense(50, input_shape=(3,), activation = 'relu'))
    model.add(Dense(25, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=1000, batch_size=11)

def test_model():
    for i in values:
        print('Our input  is : ', i, ', Output is : ', round(model.predict([format_input(i)])[0][0]))
        
def main():
    preprocess()
    train_model()
    test_model()
    
main()
    