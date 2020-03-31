# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:42:02 2020

@author: xiong
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

####obtain dataframe to labels
#A is a dataframe
#ST and END are range (from 0 to 1)
#LABEL are label. int
def get_RNN_feature_label(A,ST,END,LABEL):
    max_PN=A['PN'].max(axis = 0, skipna = True)
    features_out = []
    labels_out=[]
    for i in range(int(max_PN*ST)+1,int(max_PN*END)+1):
        print(i)
        PN_data=A.loc[A['PN']==i]
        ##fix the bug that PN particles has no data
        if len(PN_data)==0:
            continue
        PN_data=PN_data.sort_values(by=['frame'])
        F_LENGTH=20
        F_SIZE=3
        features= np.zeros((1,F_LENGTH,F_SIZE))
        temp_data=PN_data['lensf'].values.astype(float)
        temp_len=len(temp_data)
        temp_data=np.append(temp_data,PN_data['predict_A'].values.astype(float))
        temp_data=np.append(temp_data,PN_data['predict_B'].values.astype(float))
        temp_data=np.reshape(temp_data,(-1,temp_len))
        temp_data=temp_data.T
        
        if len(temp_data)>F_LENGTH:
            features[0,:,:]=temp_data[0:F_LENGTH,:]
        else:
            features[0,0:len(temp_data),:]=temp_data
        if len(features_out) == 0:
            features_out = features
            labels_out = np.append(labels_out,LABEL)
        else:
            features_out = np.append(features_out,features,axis=0)
            labels_out = np.append(labels_out,LABEL)
      
    return features_out, labels_out

#####random.shuffle
def features_labels_shuffle(train_x,train_y):
    shape_1,shape_2,shape_3=train_x.shape
    #temp arrary
    data=np.zeros((shape_1,shape_2+1,shape_3))
    data[:,0:shape_2,:]=train_x
    for i in range(shape_3):
        data[:,shape_2,i]=train_y
    #shuffle
    np.random.shuffle(data)
    
    #get return
    train_x_out=data[:,0:shape_2,:]
    train_y_out=data[:,shape_2,0]
    
    return train_x_out,train_y_out




print(tf.__version__)

root = tk.Tk()
root.withdraw()

#select dataframe files
print('select a dataframe')
dpname = filedialog.askopenfilename()
print(dpname)
A = pickle.load( open( dpname, "rb" ) )


len_A=len(A)

mean=[]
time = []
counter=0
Total_PN = A['PN'].max()
for i in range(Total_PN):
    PN_data = A.loc[A['PN']==i]
    if len(PN_data)>1:
        PN_fl = PN_data.loc[PN_data['lensf']==1]
        if len(PN_fl)>1:
            mean_fl=PN_fl['predict_B'].mean()      
            
            if mean_fl>0.5:
                mean_frame=PN_fl['frame'].mean()
                mean.append(mean_fl)
                time.append(mean_frame)
                counter=counter+1


#test_x, test_y = get_RNN_feature_label(A,0.7,1,1)

#print('select SVM model for RNN')
#dpname = filedialog.askopenfilename()
#print(dpname)
#model = pickle.load( open( dpname, "rb" ) )
#
#predictions = model.predict(test_x)