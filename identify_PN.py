# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:22:50 2020

@author: xiong
"""

import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import copy
import cv2
from sklearn.svm import SVC

#####################select dataframe files
def select_dataframe(df_name):
    print('select lensfree related to particle:'+str(df_name))
    dpname = filedialog.askopenfilename()
    print(dpname)
    
    A_lensf = pickle.load( open( dpname, "rb" ) )
    A_lensf=A_lensf.dropna(thresh=144)
    
    #select dataframe files
    print('select fluorescent image related to particle:' + str(df_name))
    dpname = filedialog.askopenfilename()
    print(dpname)
    
    A_fl = pickle.load( open( dpname, "rb" ) )
    A_fl=A_fl.dropna()
    
    return A_lensf,A_fl

def svmInit():
    clf = SVC(decision_function_shape='ovo',gamma=10, C=0.5)
    clf.probability=True
    clf.cache_size = 800
#    clf.kernel='rbf'
    clf.kernel='linear'
#    clf.kernel='poly'

    return clf

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def get_FL_feastures(A,ST,END,label=[]):
    len_A=len(A)
    B = []
    C = []
    for i in range(int(len_A*ST),int(int(len_A*END))):
        temp=A.iloc[i,0:6].values.astype(float)
#        temp_2=A.iloc[i,8:10].values.astype(float)
#        temp=np.concatenate((temp,temp_2),axis=0)
        B.append(temp)
        if label is not None:
            C.append(label)
    B=np.squeeze(B)
    C=np.squeeze(C)
    return B,C


def get_lensf_feastures(A,ST,END,label=[]):
#lensfree subimages for B
    len_A=len(A)
    B = []
    C = []
    for i in range(int(len_A*ST),int(int(len_A*END))):
        temp=A.iloc[i,9:154].values.astype(float)
        B.append(temp)
        if label is not None:
            C.append(label)            
    B=np.squeeze(B)
    C=np.squeeze(C)
    return B,C
    
##count particles based on prediction results. prediction is a 1xn arrary. count_PN is a 1xn arrary.
def counting(prediction, count_PN):
    PN_class = np.argmax(prediction)
    if PN_class==0:
        count_PN[0]=count_PN[0]+1
    elif PN_class==1:
        count_PN[1]=count_PN[1]+1
    else:
        count_PN[2]=count_PN[2]+1
    return count_PN

root = tk.Tk()
root.withdraw()

filename = 'model_lensf.sav'
model_lensf = pickle.load(open(filename, 'rb'))
filename = 'model_fl.sav'
model_fl = pickle.load(open(filename, 'rb'))

#select dp
data_lensf,data_fl= select_dataframe('ANY')


max_PN=data_fl['PN'].max(axis = 0, skipna = True)

#counting using fl, lensf and all
count_PN_fl=[0,0,0]
count_PN_lensf=[0,0,0]
count_PN_both=[0,0,0]

frame_list=np.array([[0,0,0,0]])

for i in range(int(0.05*max_PN),int(0.25*max_PN)):
    #display time taken
    if i%10==0:
        print(str(i)+'/'+str(max_PN))
    
    PN_fl_data=data_fl.loc[data_fl['PN']==i]
    PN_lensf_data=data_lensf.loc[data_lensf['PN']==i]
    #reset index
    PN_lensf_data=PN_lensf_data.reset_index(drop=True)
    
    
    #skip when no data for PN ==i
    if len(PN_fl_data) == 0 or len(PN_fl_data) == 1:
        continue
    if len(PN_lensf_data) == 0 or len(PN_lensf_data) == 1:
        continue
    frame_PN = PN_lensf_data.loc[0,'frame']
    
    ##process fl data
    PN_fl_features, ret = get_FL_feastures(PN_fl_data,0,1)
    PN_fl_predictions = model_fl.predict_proba(PN_fl_features)
    
    ##process lensf data
    PN_lensf_features, ret = get_lensf_feastures(PN_lensf_data,0,1)
    PN_lensf_predictions = model_lensf.predict_proba(PN_lensf_features)    
    
    ####count based on fl prediction
    temp = np.sum(PN_fl_predictions,axis =0)/len(PN_fl_predictions)
    count_PN_fl = counting(temp, count_PN_fl)
    
    ####count based on fl prediction
    temp = np.sum(PN_lensf_predictions,axis =0)/len(PN_lensf_predictions)
    count_PN_lensf = counting(temp, count_PN_lensf)     
    
#    ####count based on both
#when 4 class used in the training, enable the line below 
#    PN_lensf_predictions[:,2] = PN_lensf_predictions[:,2]+PN_lensf_predictions[:,3]
    PN_lensf_predictions=PN_lensf_predictions[:,0:3]
    PN_both_predictions=np.concatenate((PN_lensf_predictions, PN_fl_predictions))
    temp = np.sum(PN_both_predictions,axis =0)/len(PN_both_predictions)
    count_PN_both = counting(temp, count_PN_both) 
    
    ####add to frame list that contain start frame number of each PN
    result_temp_1 = np.array(count_PN_both)
    result_temp_2 = np.append(result_temp_1,frame_PN)
    frame_list=np.concatenate((frame_list,[result_temp_2]),axis=0)
    