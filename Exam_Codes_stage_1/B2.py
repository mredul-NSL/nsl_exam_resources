# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:33:09 2021

@author: Mredul
"""

persons = [
 {
 "name": "John",
 "age": 36,
 "country": "Norway"
 },
 {
 "name": "Bob",
 "age": 36,
 "country": "Norway"
 }
]

persons = sorted(
            sorted(
                sorted(persons, key = lambda x: x["age"]),
            key = lambda y: y["name"]),
        key =  lambda z: z["country"])

print(persons)