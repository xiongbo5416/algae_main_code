# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:34:41 2019

@author: xiong
"""

import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import cv2

root = tk.Tk()
root.withdraw()

#select videos
print('select a dp file')
dpname = filedialog.askopenfilename()
print(dpname)


data =pickle.load( open( dpname, "rb" ) )

##add perid num to dataframe
#data['period_n'] = 0
#period_i=0
#for i in data.index.values[1:]:
#    if data.loc[i,'frame']-data.loc[i-1,'frame']>1:
#        period_i=period_i+1
#    data.loc[i,'period_n']=period_i
        

#a_particle = pd.DataFrame([], columns = ['batch_diff']) 
#a_particle = data.loc[data['PN']==8]
#a_particle = a_particle.loc[a_particle['location_y']<1130]
#image_2=a_particle.loc[155,'fl_subimage']

###############https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
#########analysis all fl images
data_fl=data.loc[data['lensf']==0]
data_fl=data_fl.sort_index().reset_index(drop=True)
data_fl['x'] = np.nan
data_fl['y'] = np.nan

#logic to find the particles in the fl images
#binary,dilution
#sum
#contours the sum, find the correct position
#contour each image, and find the correct contours.

for i in data_fl.index.values:
    image=data_fl.loc[i,'fl_subimage']
    if image.max()>0:
        contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_area=0
        for c in contours:
            if cv2.contourArea(c)>10:
                if cv2.contourArea(c) > max_area:
                    box = cv2.minAreaRect(c)
                    data_fl.loc[i,'x']=box[0][0]
                    data_fl.loc[i,'y']=box[0][1]
    


#######header
#'frame' : frame_num ,'PN' :particle_List[j].PN, 'speed':speed_temp,
#'location_x':particle_List[j].position_x[-1], 'location_y':particle_List[j].position_y[-1], 
#'lensf': 1,'lensf_subimage': sub_image , 'case': particle_List[j].case

##########dimension, total, peak, x, y

#image=data.loc[249,'lensf_subimage']
#dim = (64,74)#ratio = 1.3
#image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#x=cv2.multiply(image-128+int(128/1),1)
#
#cv2.namedWindow("x", cv2.WINDOW_NORMAL) 
#cv2.imshow('x',x )
#
#image_2=data.loc[302,'fl_subimage']
##x=cv2.multiply(image-128+int(128/5),5)
#contours, hierarchy = cv2.findContours(image_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##edged = cv2.Canny(image_2, 50, 100)
##edged = cv2.dilate(edged, None, iterations=1)
##edged = cv2.erode(edged, None, iterations=1)
#box = cv2.minAreaRect(contours[0])
#bbox=cv2.boxPoints(box)
#
#
#
#cv2.namedWindow("y", cv2.WINDOW_NORMAL) 
#cv2.imshow('y',image_2 )
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
