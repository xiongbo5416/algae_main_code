# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:47:31 2020

@author: xiong
"""

import tkinter as tk
from tkinter import filedialog
import cv2


root = tk.Tk()
root.withdraw()


fig_path = filedialog.askopenfilename()

#read and show images
img = cv2.imread(fig_path,)

##resize for lensfree images
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.multiply(img,2)
img = cv2.resize(img,(1640, int(1232/1.3)))
#img = img[0:700,400:1100]

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(1):
    ##select the area or scale
    cv2.namedWindow("please select", cv2.WINDOW_NORMAL) 
    bbox = cv2.selectROI("please select",img, False)
    cv2.destroyAllWindows()
    
    ##scale bar setting
    Bar_W=2 #the width of bar, unit: pixel
    Bar_real_length=100 #the length of scale bar unit micrometer
    #for lensfree image this ratio is 1.12*2
    #for fluorescent image this ratio is 6
    #for image camptured from zhang yushan, this ratio is 6
    #for image captured by Yushan Zhang‘s microscope, for cpcc95, ratio=0.1
    Ratio=1.12*2 #the ratio between pixel to um
    
    #draw a line
#    cv2.line(img,(bbox[0],bbox[1]),(bbox[0]+int(Bar_real_length/Ratio),bbox[1]),(255,255,255),Bar_W)

#show line drawn in the image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save
cv2.imwrite(fig_path[0:-4]+'_copy.png',img)


