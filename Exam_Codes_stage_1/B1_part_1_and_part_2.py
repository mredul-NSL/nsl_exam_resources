# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 09:24:44 2021

@author: Mredul
"""
from bs4 import *
import requests
import os, os.path
from PIL import Image

  
# CREATE FOLDER
def folder_create(images,num):
    folder_name = (r"C:\Users\DELL\Desktop\Exam_Codes\image")
    os.mkdir(folder_name)
    download_images(images, folder_name, num)
  
  
def download_images(images, folder_name,num):
    count = 0
    print(f"Total {len(images)} Image Found!")
    if num > len(images):
        print("We could not find " + str(num - len(images)) + " images")
        print(len(images))
    else:
            print("we have found all images")
    if len(images) != 0:
        for i, image in enumerate(images):
            try:
                image_link = image["data-srcset"]
            except:
                try:
                    image_link = image["data-src"]
                except:
                    try:
                        image_link = image["data-fallback-src"]
                    except:
                        try:
                            image_link = image["src"]
                        except:
                            pass

            try:
                r = requests.get(image_link).content
                try:
  
                    r = str(r, 'utf-8')
  
                except UnicodeDecodeError:
  
                    with open(f"{folder_name}/images{i+1}", "wb+") as f:
                        f.write(r)
                    count += 1
            except:
                pass
  
        if count == len(images):
            print("All Images Downloaded!")
              
        else:
            print(f"Total {count} Images Downloaded Out of {len(images)}")
  

def resize_image():
    imgs = []
    path = r"C:\Users\DELL\Desktop\Exam_Codes\image"
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(Image.open(os.path.join(path,f)))
        
    for img in imgs:
        new_img = img.resize((224,224))
        new_img.save(r"C:\Users\DELL\Desktop\Exam_Codes\preprocessed")
    print("All image resized")

def main(url):
    r = requests.get(url)
    number_of_images = int(input())
    soup = BeautifulSoup(r.text, 'html.parser')
    images = soup.findAll('img')
    folder_create(images, number_of_images)
    resize_image()
  
  
url = "https://www.prothomalo.com/"
  
main(url)