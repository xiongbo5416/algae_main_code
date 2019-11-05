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
frame_i=1
font = cv2.FONT_HERSHEY_SIMPLEX
for f in glob.glob(os.path.join(images_folder, "*.bmp")):
    frame_i=frame_i+1
    print(f)
    if frame_i <30*60*10:
        continue
    if frame_i >30*60*15:
        break
    
    img = cv2.imread(f)
    img_array.append(img)
    height, width, layers = img.shape
    size = (width,height)
    cv2.putText(img, str(frame_i), (100,100), font, 2, (255), 2, cv2.LINE_AA)
    #frame_i=frame_i+1
    #print(f)
    
out = cv2.VideoWriter('16_20min.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
    
out.release()


 
