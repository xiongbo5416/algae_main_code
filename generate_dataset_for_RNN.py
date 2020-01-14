# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:34:41 2019

@author: xiong
"""

#input 
#1.results.p in each trail
#2.model trained by SVM

#output?


import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import copy
import cv2
from sklearn.svm import SVC

def get_FL_feastures(A,ST,END):
    len_A=len(A)
    B = []
    for i in range(int(len_A*ST),int(int(len_A*END))):
        temp=A.iloc[i,0:6].values.astype(float)
#        temp_2=A.iloc[i,8:10].values.astype(float)
#        temp=np.concatenate((temp,temp_2),axis=0)
        B.append(temp)
    B=np.squeeze(B)
    return B

def get_lensf_feastures(A,ST,END):
#lensfree subimages for B
    len_A=len(A)
    B = []
    for i in range(int(len_A*ST),int(int(len_A*END))):
        temp=A.iloc[i,11:156].values.astype(float)
        B.append(temp)
    B=np.squeeze(B)
    return B

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
#read data from results
data =pickle.load( open( dpname, "rb" ))

#for hog
FL_HEIGHT=120
FL_WIDTH=70
THRESHOLD_FL=4
STRIDE_WINDOW=16
WIDE_WINDOW=4*16
HEIGHT_WINDOW=4*16

#######analyze each particles
#logic flow
# loops for each particles
# remove not realiable data
max_PN=data['PN'].max(axis = 0, skipna = True)


c_temp=[]
data_fl=pd.DataFrame([], columns = ['F_dimension','F_length','F_width','F_average','F_peak','F_total','position_x','position_y','frame','lensf']) 
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
    PN_data_fl = pd.DataFrame([], columns = ['F_dimension','F_length','F_width','F_average','F_peak','F_total','position_x','position_y','frame','lensf']) 
    PN_data_fl['period_n']=PN_data.loc[PN_data['lensf']==0]['period_n']
    PN_data_fl['PN']=PN_data.loc[PN_data['lensf']==0]['PN']
    PN_data_fl['frame']=PN_data.loc[PN_data['lensf']==0]['frame']
    PN_data_fl['lensf']=PN_data.loc[PN_data['lensf']==0]['lensf']

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
                    PN_data_fl.loc[h,'F_dimension']= cv2.contourArea(c_temp)
                    PN_data_fl.loc[h,'F_length']= box[1][0]
                    PN_data_fl.loc[h,'F_width']= box[1][1]
                    PN_data_fl.loc[h,'F_peak']= img_temp.max()
                    PN_data_fl.loc[h,'F_average']=sum(sum(img_temp))/PN_data_fl.loc[h,'F_dimension']
                    PN_data_fl.loc[h,'F_total']=sum(sum(img_temp))
                    PN_data_fl.loc[h,'position_x']=box[0][0]
                    PN_data_fl.loc[h,'position_y']=box[0][1]
                else:
                    PN_data_fl.loc[h,'F_dimension']= 0
                    PN_data_fl.loc[h,'F_length']= 0
                    PN_data_fl.loc[h,'F_width']= 0
                    PN_data_fl.loc[h,'F_peak']= 0
                    PN_data_fl.loc[h,'F_average']=0
                    PN_data_fl.loc[h,'F_total']=0
                    PN_data_fl.loc[h,'position_x']=-1
                    PN_data_fl.loc[h,'position_y']=-1
    

    #append results to fl output
    data_fl=pd.concat([data_fl,PN_data_fl])   
                    
# process lensfree_sub image data
data_lensf= data.loc[data['lensf']==1]
#data_lensf = data_lensf.loc[data_lensf['case']!=6]
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
        


##get FL features
data_fl=data_fl.dropna()
FL_features = get_FL_feastures(data_fl,0,1)
##predict FL subimages using FL features
#select a trained SVM
print('select SVM model for FL')
dpname = filedialog.askopenfilename()
print(dpname)
model_fl = pickle.load( open( dpname, "rb" ) )
#test using svm
FL_predictions = model_fl.predict_proba(FL_features)

data_fl['predict_A']=FL_predictions[:,0]
data_fl['predict_B']=FL_predictions[:,1]


##########get lensfree freatures
data_lensf=data_lensf.dropna(thresh=144)
lensf_features = get_lensf_feastures(data_lensf,0,1)
##predict FL subimages using FL features
#select a trained SVM
print('select SVM model for lensf')
dpname = filedialog.askopenfilename()
print(dpname)
model_lensf = pickle.load( open( dpname, "rb" ) )
#test using svm
lensf_predictions = model_lensf.predict_proba(lensf_features)
data_lensf['predict_A']=lensf_predictions[:,0]
data_lensf['predict_B']=lensf_predictions[:,1]


#####generate final dataframe
RNN_train_1 = pd.DataFrame([], columns = ['lensf','PN','frame','predict_A','predict_B'])
RNN_train_1['lensf']=data_fl['lensf']
RNN_train_1['PN']=data_fl['PN']
RNN_train_1['frame']=data_fl['frame']
RNN_train_1['predict_A']=data_fl['predict_A']
RNN_train_1['predict_B']=data_fl['predict_B']

RNN_train_2 = pd.DataFrame([], columns = ['lensf','PN','frame','predict_A','predict_B'])
RNN_train_2['lensf']=data_lensf['lensf']
RNN_train_2['PN']=data_lensf['PN']
RNN_train_2['frame']=data_lensf['frame']
RNN_train_2['predict_A']=data_lensf['predict_A']
RNN_train_2['predict_B']=data_lensf['predict_B']

RNN_train=pd.concat([RNN_train_1,RNN_train_2])

#save
pickle.dump( RNN_train, open( "RNN_train.p", "wb" ) )
