# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:31:43 2019

@author: xiong
"""
import tkinter as tk
from tkinter import filedialog
import glob
import os
import cv2

root = tk.Tk()
root.withdraw()
img_array = []
images_folder = filedialog.askdirectory()
for f in glob.glob(os.path.join(images_folder, "*.jpg")):
    img = cv2.imread(f)
    #img = img[0:1000,0:1300,:]
    img_array.append(img)
    height, width, layers = img.shape
    size = (width,height)
    print(f)
    
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
    
out.release()


 
