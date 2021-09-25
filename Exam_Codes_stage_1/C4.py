# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 09:47:45 2021

@author: Mredul
"""

class Layer:
     def __init__(self, name):
        self.name = name
 
     def __call__(self):
        print("layer " + self.name + " is called")
        pass

layer = Layer("custom layer name")
print(layer())



#y = layer("image")