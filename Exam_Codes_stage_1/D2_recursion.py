# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 12:18:29 2021

@author: Mredul
"""

import os

def visit_directory(base_path):

    try:
        folders = os.listdir(str(base_path))
    except:
        pass
    
    for folder in folders:
        print(folder)
        files = visit_directory(base_path + '/' + folder)
        for file in files:
            print(file)
            

visit_directory('test/')
