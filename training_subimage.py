# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:12:37 2019

@author: xiong
"""

import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.svm import SVC


def svmInit(C=5, gamma=0.50625):
    clf = SVC(gamma='auto')
    clf.probability=True
    clf.kernel='rbf'
    #clf.kernel='poly'
    clf.C=5
  
    return clf

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def get_FL_feastures(A,ST,END):
    len_A=len(A)
    B = []
    for i in range(int(len_A*ST),int(int(len_A*END))):
        temp=A.iloc[i,1:5].values.astype(float)
        temp_2=A.iloc[i,8:10].values.astype(float)
        temp=np.concatenate((temp,temp_2),axis=0)
        B.append(temp)
    B=np.squeeze(B)
    return B

root = tk.Tk()
root.withdraw()

############select dataframe files
#select dataframe files
print('select lensfree related to particle green algae')
dpname = filedialog.askopenfilename()
print(dpname)

A_lensf = pickle.load( open( dpname, "rb" ) )
A_lensf=A_lensf.dropna(thresh=144)

#select dataframe files
print('select fluorescent image related to particle green algae')
dpname = filedialog.askopenfilename()
print(dpname)

A_fl = pickle.load( open( dpname, "rb" ) )
A_fl=A_fl.dropna()

#select dataframe files
print('select lensfree related to particle 10um beads')
dpname = filedialog.askopenfilename()
print(dpname)

B_lensf = pickle.load( open( dpname, "rb" ) )
B_lensf =B_lensf.dropna(thresh=144)

#select dataframe files
print('select fluorescent image related to 10um beads')
dpname = filedialog.askopenfilename()
print(dpname)

B_fl = pickle.load( open( dpname, "rb" ) )
B_fl=B_fl.dropna()


############ training parameters init
model_lensf = svmInit()
Features_lensf_train = []
labels_lensf_train=[]


model_fl = svmInit()
Features_fl_train = []
labels_fl_train=[]

#########get training data
len_A_lensf=len(A_lensf)
len_B_lensf=len(B_lensf)
len_A_fl=len(A_fl)
len_B_fl=len(B_fl)

#get Features_lensf_train
for i in range(int(len_A_lensf/2)):
    temp=A_lensf.iloc[i,11:156].values.astype(float)
    Features_lensf_train.append(temp)
    labels_lensf_train.append(1)
    
for i in range(int(len_B_lensf/2)):
    temp=B_lensf.iloc[i,11:156].values.astype(float)
    Features_lensf_train.append(temp)
    labels_lensf_train.append(2)
    
Features_lensf_train = np.squeeze(Features_lensf_train)
labels_lensf_train=np.squeeze(labels_lensf_train)

#get Features_FL_train
for i in range(int(len_A_fl/2)):
    temp=A_fl.iloc[i,1:5].values.astype(float)
    temp_2=A_fl.iloc[i,8:10].values.astype(float)
    temp=np.concatenate((temp,temp_2),axis=0)
    Features_fl_train.append(temp)
    labels_fl_train.append(1)
    
for i in range(int(len_B_fl/2)):
    temp=B_fl.iloc[i,1:5].values.astype(float)
    temp_2=B_fl.iloc[i,8:10].values.astype(float)
    temp=np.concatenate((temp,temp_2),axis=0)
    Features_fl_train.append(temp)
    labels_fl_train.append(2)
    
Features_fl_train = np.squeeze(Features_fl_train)
labels_fl_train=np.squeeze(labels_fl_train)

#train model
model_lensf.fit(Features_lensf_train, labels_lensf_train)
model_fl.fit(Features_fl_train, labels_fl_train)

############### test 
#lensfree subimages for A
A_lensf_test = []
for i in range(int(len_A_lensf/2),int(len_A_lensf)):
    temp=A_lensf.iloc[i,11:156].values.astype(float)
    A_lensf_test.append(temp)
A_lensf_test=np.squeeze(A_lensf_test)

#fluorescent subimages for A 
A_fl_test = []
for i in range(int(len_A_fl/2),int(len_A_fl)):
    temp=A_fl.iloc[i,1:5].values.astype(float)
    temp_2=A_fl.iloc[i,8:10].values.astype(float)
    temp=np.concatenate((temp,temp_2),axis=0)
    A_fl_test.append(temp)
A_fl_test=np.squeeze(A_fl_test)

#lensfree subimages for B
B_lensf_test = []
for i in range(int(len_B_lensf/2),int(len_B_lensf)):
    temp=B_lensf.iloc[i,11:156].values.astype(float)
    B_lensf_test.append(temp)
B_lensf_test=np.squeeze(B_lensf_test)

#fluorescent subimages for B 
B_fl_test = []
for i in range(int(len_B_fl/2),int(len_B_fl)):
    temp=B_fl.iloc[i,1:5].values.astype(float)
    temp_2=B_fl.iloc[i,8:10].values.astype(float)
    temp=np.concatenate((temp,temp_2),axis=0)
    B_fl_test.append(temp)
B_fl_test=np.squeeze(B_fl_test)

#test using svm
A_lensf_predictions = model_lensf.predict_proba(A_lensf_test)
A_lensf_predictions[:,0]=A_lensf_predictions[:,0]
A_lensf_predictions[:,1]=A_lensf_predictions[:,1]
A_lensf_predictions=np.rint(A_lensf_predictions)
print( 'sum(A_lensf_predictions)'+ str(sum(A_lensf_predictions)))
sum(A_lensf_predictions)
 
B_lensf_predictions = model_lensf.predict_proba(B_lensf_test)
B_lensf_predictions[:,0]=B_lensf_predictions[:,0]
B_lensf_predictions[:,1]=B_lensf_predictions[:,1]
B_lensf_predictions=np.rint(B_lensf_predictions)
print( 'sum(B_lensf_predictions)'+ str(sum(B_lensf_predictions)))
sum(B_lensf_predictions)

A_fl_predictions = model_fl.predict_proba(A_fl_test)
A_fl_predictions=np.rint(A_fl_predictions)
print( 'sum(A_fl_predictions)')
sum(A_fl_predictions)

B_fl_predictions = model_fl.predict_proba(B_fl_test)
B_fl_predictions=np.rint(B_fl_predictions)
print( 'sum(B_fl_predictions)')
sum(B_fl_predictions)

####save svm model
pickle.dump( model_lensf, open( "model_lensf.sav", "wb" ) )
pickle.dump( model_fl, open( "model_fl.sav", "wb" ) )

