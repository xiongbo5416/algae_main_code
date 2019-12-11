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
import copy
import cv2

root = tk.Tk()
root.withdraw()

#select videos
print('select a dp file')
dpname = filedialog.askopenfilename()
print(dpname)


data =pickle.load( open( dpname, "rb" ) )
c_temp=[]
data_fl=pd.DataFrame([], columns = ['fl_img','dimension','length','width','average','peak','total','position_x','position_y']) 
#######analyze each particles
#logic flow
# loops for each particles
# remove not realiable data
# process fl_sub image data
for i in range(1):
    PN_data=data.loc[data['PN']==3]
    
    #filterout not reliable data
#    PN_data=PN_data.loc[PN_data['location_y']<1200]
#    img=PN_data.loc[334,'fl_subimage']
    #process fl_subimage
    PN_data_fl = pd.DataFrame([], columns = ['fl_img','dimension','length','width','average','peak','total','position_x','position_y']) 
    PN_data_fl['fl_img']=PN_data.loc[PN_data['lensf']==0]['fl_subimage']
    PN_data_fl['period_n']=PN_data.loc[PN_data['lensf']==0]['period_n']
    ## remove false detection
    if len(PN_data_fl)==0:
        continue
    
    #bgd subtract
    p_min=min(PN_data_fl['period_n'])
    p_max=max(PN_data_fl['period_n'])
    
    #loop for each period in a particle
    for j in range(p_min,p_max+1):
        #get the index of rows that contain fl image
        h_index=PN_data_fl.loc[PN_data_fl['period_n']==j].index.values.astype(int)
        #fix a bug, when period number jumps
        if len(h_index)==0:
            continue
        #generate a mask that can enable the useful information in each fl imgs
        mask_fl= PN_data_fl.loc[h_index[0],'fl_img']*0
        if len(h_index)>3:#at least 3 fl image captured in a period
            #####get the integration of a few fl images in one period
            mask = PN_data_fl.loc[h_index[0],'fl_img']*0
            for h in h_index:
                mask_temp = PN_data_fl.loc[h,'fl_img']
                #Binary for each fl image
                ret,mask_temp = cv2.threshold(mask_temp,20,1,cv2.THRESH_BINARY)
                mask= mask+ mask_temp
            
            ###########find the overlap area that has the biggest area. this area could be a result of tracked particles
            ret,mask_temp = cv2.threshold(mask,len(h_index)-1,1,cv2.THRESH_BINARY)         
            #get the roughly area of the target particle in a fl image
            contours, hierarchy = cv2.findContours(mask_temp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            max_area=-1
            #get the positon of integration that has the largest area
            for c in contours:
                if cv2.contourArea(c) > max_area:
                    box = cv2.minAreaRect(c)
                    x_temp=box[0][0]
                    y_temp=box[0][1]
                    max_area = cv2.contourArea(c)
                    
            ##########generate a mask that can enable the useful information in each fl imgs
            if max_area>-1:
                contours_2, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                k=0
                for c in contours_2:
                    if max(c[:,0,0])>x_temp and min(c[:,0,0])<x_temp:
                        if max(c[:,0,1])>y_temp and min(c[:,0,1])<y_temp:
                            cv2.drawContours(mask_fl, contours_2, k, (255),cv2.FILLED)
                    k=k+1
            
            
            
            
            ###########get refined fl imge and save it into 'fl_img'
            for h in h_index:
                img_temp = PN_data_fl.loc[h,'fl_img']
                #apply a mask to fl subimage to fiter out the useless particles
                img_temp  = cv2.bitwise_and(img_temp ,img_temp ,mask = mask_fl)
                contours_3, hierarchy = cv2.findContours(img_temp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                #find the max contours in contours_3
                area_max=-1
                for c in contours_3:
                    if len(c)>area_max:
                        c_temp=c
                        area_max=len(c)
                
                #when a particle is found write useful information in a PN_data_fl
                if area_max>-1:
                    box = cv2.minAreaRect(c_temp)
                    PN_data_fl.loc[h,'dimension']= cv2.contourArea(c_temp)
                    PN_data_fl.loc[h,'length']= box[1][0]
                    PN_data_fl.loc[h,'width']= box[1][1]
                    PN_data_fl.loc[h,'peak']= img_temp.max()
                    PN_data_fl.loc[h,'average']=sum(sum(img_temp))/PN_data_fl.loc[h,'dimension']
                    PN_data_fl.loc[h,'total']=sum(sum(img_temp))
                    PN_data_fl.loc[h,'position_x']=box[0][0]
                    PN_data_fl.loc[h,'position_y']=box[0][1]
    
    #append results to fl output
    data_fl=pd.concat([data_fl,PN_data_fl])   
                    
#data = pd.concat([data, data_fl], axis=1, sort=False)
                    
                
#                max_dis=10000
#                for c in contours:
#                    if cv2.contourArea(c) > max_area/2:
#                        distance_temp= (box[0][0]-x_temp)*(box[0][0]-x_temp)+(box[0][1]-y_temp)*(box[0][1]-y_temp)
#                        if distance_temp < max_dis:
#                            box = cv2.minAreaRect(c)
#                            c_temp=c
#                            max_dis=distance_temp
#                if distance_temp<300:
#                   box = cv2.minAreaRect(c_temp)
#                   PN_data_fl.loc[h,'dimension']= cv2.contourArea(c)
                    


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

#data_fl=data.loc[data['lensf']==0]
#data_fl=data_fl.sort_index().reset_index(drop=True)
#data_fl['x'] = np.nan
#data_fl['y'] = np.nan
#
##logic to find the particles in the fl images
##binary,dilution
##sum
##contours the sum, find the correct position
##contour each image, and find the correct contours.
#
#for i in data_fl.index.values:
#    image=data_fl.loc[i,'fl_subimage']
#    if image.max()>0:
#        contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#        max_area=0
#        for c in contours:
#            if cv2.contourArea(c)>10:
#                if cv2.contourArea(c) > max_area:
#                    box = cv2.minAreaRect(c)
#                    data_fl.loc[i,'x']=box[0][0]
#                    data_fl.loc[i,'y']=box[0][1]
#    
#
#image_2=data.loc[104,'fl_subimage']
#data_fl['y'].describe()
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
