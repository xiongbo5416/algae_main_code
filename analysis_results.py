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

###hog
def get_hog() : 
    winSize = (WIDE_WINDOW,HEIGHT_WINDOW)
    blockSize = (STRIDE_WINDOW,STRIDE_WINDOW)
    blockStride = (STRIDE_WINDOW,STRIDE_WINDOW)
    cellSize = (STRIDE_WINDOW,STRIDE_WINDOW)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    
    
    
root = tk.Tk()
root.withdraw()

#select videos
print('select a dp file')
dpname = filedialog.askopenfilename()
print(dpname)

FL_HEIGHT=160
FL_WIDTH=80
THRESHOLD_FL=3
STRIDE_WINDOW=16
WIDE_WINDOW=4*16
HEIGHT_WINDOW=4*16

data =pickle.load( open( dpname, "rb" ) )



c_temp=[]
data_fl=pd.DataFrame([], columns = ['dimension','length','width','average','peak','total','position_x','position_y']) 
#######analyze each particles
#logic flow
# loops for each particles
# remove not realiable data
max_PN=data['PN'].max(axis = 0, skipna = True)

# process fl_sub image data
for i in range(max_PN+1):
    #display time taken
    if i%10==0:
        print(str(i)+'/'+str(max_PN))
    
    PN_data=data.loc[data['PN']==i]
    #remove the unreliable particles
    #remove low speed particles
    speed_mean = PN_data['speed_y'].mean(axis = 0, skipna = True)
    if speed_mean<1:
        data=data.drop(PN_data.index.values.astype(int))
        continue
    #remove the particles which closed to the edge of channel
    location_x_mean = PN_data['location_x'].mean(axis = 0, skipna = True)
    if location_x_mean>1220:
        data=data.drop(PN_data.index.values.astype(int))
        continue

    
#    #filterout not reliable data
##    PN_data=PN_data.loc[PN_data['location_y']<1200]
##    img=PN_data.loc[334,'fl_subimage']
#    #process fl_subimage
    PN_data_fl = pd.DataFrame([], columns = ['dimension','length','width','average','peak','total','position_x','position_y']) 
    PN_data_fl['period_n']=PN_data.loc[PN_data['lensf']==0]['period_n']
    PN_data_fl['PN']=PN_data.loc[PN_data['lensf']==0]['PN']
    ## remove false detection
    if len(PN_data)==0:
        continue
    
    #bgd subtract
    p_min=min(PN_data['period_n'])
    p_max=max(PN_data['period_n'])
    
    #loop for each period in a particle
    for j in range(p_min,p_max+1):
        #get the dataframe that contain fl image
        temp_data=PN_data.loc[PN_data['period_n']==j]
        temp_FL_img=temp_data.loc[temp_data['lensf']==0]
        
        #remove the bad fl image, drop the row that contain bad fl image in the dataframes
        h_index=temp_FL_img.index.values.astype(int)
        for h in h_index:
            mask_temp = temp_FL_img.loc[h,'fl_subimage']
            if mask_temp.shape[0] != FL_HEIGHT or mask_temp.shape[1] != FL_WIDTH:
                temp_FL_img=temp_FL_img.drop(h)
                data=data.drop(h)
                PN_data_fl=PN_data_fl.drop(h)
                
                
        h_index=temp_FL_img.index.values.astype(int)
        #fix a bug, when period number jumps
        if len(h_index)==0:
            continue
        #generate a mask that can enable the useful information in each fl imgs
        mask_fl= np.zeros((FL_HEIGHT,FL_WIDTH),np.uint8)
        if len(h_index)>3:#at least 3 fl image captured in a period
            #####get the integration of a few fl images in one period
            mask_overlap = np.zeros((FL_HEIGHT,FL_WIDTH),np.uint8)
            for h in h_index:
                mask_temp = temp_FL_img.loc[h,'fl_subimage']
                if mask_temp.shape[0] == FL_HEIGHT and mask_temp.shape[1] == FL_WIDTH:
                    #Binary for each fl image
                    ret,mask_temp = cv2.threshold(mask_temp,THRESHOLD_FL,1,cv2.THRESH_BINARY)
                    mask_overlap= mask_overlap+ mask_temp
            
            ###########find the overlap area that has the biggest area. this area could be a result of tracked particles
            ret,mask_temp = cv2.threshold(mask_overlap,len(h_index)-1,1,cv2.THRESH_BINARY)         
            #get the roughly area of the target particle in a fl image
            contours, hierarchy = cv2.findContours(mask_temp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            max_area=-1
            #get the positon of integration that has the largest area
            for c in contours:
                if cv2.contourArea(c) > max_area:
                    box = cv2.minAreaRect(c)
                    x_target=box[0][0]
                    y_target=box[0][1]
                    max_area = cv2.contourArea(c)
                    
            ##########generate a mask that can enable the useful information in each fl imgs
            if max_area>-1:
                contours_2, hierarchy = cv2.findContours(mask_overlap,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                k=0
                for c in contours_2:
                    if max(c[:,0,0])>x_target and min(c[:,0,0])<x_target:
                        if max(c[:,0,1])>y_target and min(c[:,0,1])<y_target:
                            cv2.drawContours(mask_fl, contours_2, k, (255),cv2.FILLED)
                    k=k+1
            
            
            
            
            ###########get refined fl imge and save it into 'fl_img'
            for h in h_index:
                img_temp = temp_FL_img.loc[h,'fl_subimage']
                if img_temp.shape[0] != FL_HEIGHT or img_temp.shape[1] != FL_WIDTH:
                    continue
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
                else:
                    PN_data_fl.loc[h,'dimension']= 0
                    PN_data_fl.loc[h,'length']= 0
                    PN_data_fl.loc[h,'width']= 0
                    PN_data_fl.loc[h,'peak']= 0
                    PN_data_fl.loc[h,'average']=0
                    PN_data_fl.loc[h,'total']=0
                    PN_data_fl.loc[h,'position_x']=-1
                    PN_data_fl.loc[h,'position_y']=-1
    
    #append results to fl output
    data_fl=pd.concat([data_fl,PN_data_fl])   
                    
# process fl_sub image data
data_lensf= data.loc[data['lensf']==1]
data_lensf = data_lensf.loc[data_lensf['case']!=6]
data_lensf['contrast'] =np.NAN
for i in range(144):
    data_lensf['hog'+str(i)] =np.NAN
    
h_index=data_lensf.index.values.astype(int)

i_time=-1
all_time=len(h_index)
for h in h_index:
    #display time taken
    i_time=i_time+1
    if i_time%1000==0:
        print(str(i_time)+'/'+str(all_time))
    
    img_temp=data_lensf.loc[h,'lensf_subimage']
    if img_temp.shape[0] != 1:
        #get and save hog
        hog = get_hog()
        #resize img to 64x64. 64x64 is big enough
        img_temp=img_temp[16:-16,16:-16]
        
        descriptor = hog.compute(img_temp,(STRIDE_WINDOW,STRIDE_WINDOW),(0,0)) 
        descriptor=descriptor[:,0]
        descriptor = descriptor.ravel() 
        data_lensf.iloc[i_time,12:156]=descriptor
    
        ##get contrast of each subimage
        img_temp_2=img_temp.astype(np.int16)
        img_temp_2=img_temp_2-128
        img_temp_2=np.square(img_temp_2) 
        contrast=img_temp_2.sum()
        data_lensf.loc[h,'contrast']=contrast
        
pickle.dump( data_lensf, open( "data_lensf.p", "wb" ) )
pickle.dump( data_fl, open( "data_fl.p", "wb" ) )
    
