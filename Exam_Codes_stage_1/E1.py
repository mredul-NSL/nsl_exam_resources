# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:58:12 2021

@author: Mredul
"""

import numpy as np
import PIL
from PIL import ImageOps
from PIL import Image


image_file = PIL.Image.open("E1.png")
image_file = ImageOps.grayscale(image_file) 
image_array = np.array(image_file)

image_array = 255 - image_array

mean = np.mean(image_array)

row_sum = np.sum(image_array , axis=1)

j = 0
image_rows = []
for i in range(1, image_array.shape[0]):
    if row_sum[i] == 0:
        image_rows.append(image_array[j:i, :])
        j=i
image_rows.append(image_array[j:, :])

     
row_number = 0
for i in range(len(image_rows)):
    if np.sum(image_rows[i]) != 0:
        image = Image.fromarray(image_rows[i])
        image = image.convert('RGB')
        image.save('img' + str(row_number) + '.png')
        col_sum = np.sum(image_rows[i], axis=0)
        row_number += 1


row_number = 4
for number in range(0,4):
    image_file = PIL.Image.open("img"+str(number)+".png")
    image_file = ImageOps.grayscale(image_file) 
    image_array = np.array(image_file)
    
    mean = np.mean(image_array)
    
    col_sum = np.sum(image_array , axis = 0)
    
    j = 0
    image_cols = []
    for i in range(1, image_array.shape[1]):
        if col_sum[i] == 0:
            image_cols.append(image_array[:, j:i])
            j=i
    image_cols.append(image_array[:, j:])
    
         
    
    for i in range(len(image_cols)):
        if np.sum(image_cols[i]) != 0:
            image = Image.fromarray(image_cols[i])
            image = image.convert('RGB')
            image.save('img' + str(row_number) + '.png')
            #print(np.array(image))
            row_number += 1