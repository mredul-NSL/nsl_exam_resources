# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:03:41 2021

@author: DELL
"""

class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
# child class
class FlowerImage(Image):
    def __init__(self, width, height, flower_name):
        super().__init__(width, height)
        pass
