# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:21:22 2019

@author: xiong
"""

import tkinter as tk
from tkinter import filedialog
import cv2
import glob
import os
import numpy as np


root = tk.Tk()
root.withdraw()


###########for single image
#fig_path = filedialog.askopenfilename()
#
#img = cv2.imread(fig_path,)
#grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

################### for imges in a folder
images_folder = filedialog.askdirectory()
i = 0
for f in glob.glob(os.path.join(images_folder, "*.jpg")):
    if i == 0:        
        frame = cv2.imread(f)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_f=frame.astype(np.float)
        i=i+1
    else:
        frame = cv2.imread(f)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_f=frame_f+frame
        i=i+1

frame_f=frame_f/i
img=frame_f.astype(np.uint8)

dimension=[0,0,0,0,0,0,0,0,0,0]
R=[0,0,0,0,0,0,0,0,0,0]

for  j in range(6):
    cv2.namedWindow("please select", cv2.WINDOW_NORMAL) 
    bbox = cv2.selectROI("please select",img, True)
    cv2.destroyAllWindows()
    print("    <box top='" + str(bbox[1]) + "' left ='" + str(bbox[0]) + "' width ='" + str(bbox[2]) + "' height ='" + str(bbox[3]) + "'/>" )
    
    WIDTH=60
    HEIGHT=60
    img_cut = img[int(bbox[1]+bbox[3]/2-HEIGHT/2):int(bbox[1]+bbox[3]/2+HEIGHT/2),int(bbox[0]+bbox[2]/2-WIDTH/2):int(bbox[0]+bbox[2]/2+WIDTH/2)]
    
    ######get bgd and top value of cut_img
    Mask_out= img_cut*0+255
    Mask_out[15:45,15:45]=0
    bgd_img= cv2.bitwise_and(img_cut, img_cut, mask=Mask_out)
    bgd_img=bgd_img*1.0
    bgd=sum(sum(bgd_img))/sum(sum(Mask_out/255))
    top= img_cut.max()
    
    ####get thorshold
    thor=top*0.2+bgd*0.8
    ret,thresh1 = cv2.threshold(img_cut,thor,255,cv2.THRESH_BINARY)
    
    #get deminsion
    dimension[j]= sum(sum(thresh1/255))
    dimension[j]=dimension[j]*36
    R[j]=pow(dimension[j]/3.14,0.5)
    print(dimension[j])
    print(R[j])
    
    cv2.imshow('image_cut',img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()