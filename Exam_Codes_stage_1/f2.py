# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:41:42 2021

@author: Mredul
"""
import os
import cv2
import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Add, Flatten, InputLayer
from tensorflow.nn import relu
from tensorflow.keras.optimizers import Adam

training_data = [] 
x = []
y = []
new_img_size = 100

def preprocess():
    CATEGORIES = ["Dog", "Cat"]
    DATADIR = "D:\cat-dog-data\PetImages"
    for category in CATEGORIES:  
        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):  
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (new_img_size,new_img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    
   
    for features, label in training_data:
        x.append(features)
        y.append(label)
    
        


    
def train_model():
    
    input1 = Input(shape = (new_img_size,new_img_size, 1), name = "input_1")
    input2 = Input(shape = (new_img_size,new_img_size, 1), name = "input_2")
    
    conv_x = Conv2D(filters=4,kernel_size=3,name='conv_x')(input1)
    relu_x = relu(conv_x)
    
    conv_y = Conv2D(filters=4,kernel_size=3,name='conv_y')(input2)
    relu_y = relu(conv_y)
    
    merge = Add()([relu_x,relu_y])
    
    conv_z = Conv2D(filters=16,kernel_size=3,name='conv_z')(merge)
    relu_z = relu(conv_z)
    
    flatten = Flatten()(relu_z)
    
    output = Dense(1,activation='relu')(flatten)
    
    model  = tf.keras.Model([input1,input2],output)
    
    print(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])
    y_train = np.array(y)
    model.fit(x, y_train, epochs=10, batch_size=8)
    model.save("f2.model")


def test_model():
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    final = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    model = tf.keras.models.load_model("f2.model")

    prediction = model.predict([prepare('test.jpg')])
    print(prediction)  
    print(CATEGORIES[int(prediction[0][0])])

def main():
    preprocess()
    train_model()
    test_model()
    
main()
