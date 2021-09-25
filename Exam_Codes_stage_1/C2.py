# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 10:52:00 2021

@author: Mredul
"""
class LaptopBrand(object):
    def __init__(self):
        self.name = {0 : "Lenovo" , 1 : "Dell", 2 :"Hp"}
        
    def __getitem__(self, item):
         return self.LaptopBrand[item]



print(LaptopBrand[0])