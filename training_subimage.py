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
    

############select dataframe files

#####A is 243, B is 95, C is 10um beads, D is other non-fluorescent particles
######################select dataframe files

A_lensf,A_fl= select_dataframe('243')
##drop zeros
A_fl = A_fl.replace(0, np.nan)
A_fl=A_fl.dropna(thresh=7)
A_fl = A_fl.replace(np.nan,0)

B_lensf,B_fl= select_dataframe('95')
##drop zeros
B_fl = B_fl.replace(0, np.nan)
B_fl=B_fl.dropna(thresh=7)
B_fl = B_fl.replace(np.nan,0)

C_lensf,C_fl= select_dataframe('10um')

D_lensf,D_fl= select_dataframe('BBM')



############ training parameters init
model_lensf = svmInit()
Features_lensf_train = []
labels_lensf_train=[]


model_fl = svmInit()
Features_fl_train = []
labels_fl_train=[]

#########get training data


#get Features_lensf_train
temp_f, temp_l = get_lensf_feastures(A_lensf,0,0.1,1)
Features_lensf_train=temp_f
labels_lensf_train=temp_l

temp_f, temp_l = get_lensf_feastures(B_lensf,0,0.1,2)
Features_lensf_train=np.append(Features_lensf_train,temp_f,axis=0)
labels_lensf_train=np.append(labels_lensf_train,temp_l,axis=0)

temp_f, temp_l = get_lensf_feastures(C_lensf,0,0.02,3)
Features_lensf_train=np.append(Features_lensf_train,temp_f,axis=0)
labels_lensf_train=np.append(labels_lensf_train,temp_l,axis=0)

temp_f, temp_l = get_lensf_feastures(D_lensf,0,0.1,4)
Features_lensf_train=np.append(Features_lensf_train,temp_f,axis=0)
labels_lensf_train=np.append(labels_lensf_train,temp_l,axis=0)

#get Features_fl_train
temp_f, temp_l = get_FL_feastures(A_fl,0,0.1,1)
Features_fl_train=temp_f
labels_fl_train=temp_l

temp_f, temp_l = get_FL_feastures(B_fl,0,0.1,2)
Features_fl_train=np.append(Features_fl_train,temp_f,axis=0)
labels_fl_train=np.append(labels_fl_train,temp_l,axis=0)

temp_f, temp_l = get_FL_feastures(C_fl,0,0.02,3)
Features_fl_train=np.append(Features_fl_train,temp_f,axis=0)
labels_fl_train=np.append(labels_fl_train,temp_l,axis=0)

temp_f, temp_l = get_FL_feastures(D_fl,0,0.1,3)
Features_fl_train=np.append(Features_fl_train,temp_f,axis=0)
labels_fl_train=np.append(labels_fl_train,temp_l,axis=0)

#train model
model_lensf.fit(Features_lensf_train, labels_lensf_train)
model_fl.fit(Features_fl_train, labels_fl_train)

####save svm model
pickle.dump( model_lensf, open( "model_lensf.sav", "wb" ) )
print('lensf model trained')
pickle.dump( model_fl, open( "model_fl.sav", "wb" ) )

#test
temp_f, temp_l = get_FL_feastures(B_fl,0.2,1)
#C_fl_predictions = model_fl.predict(temp_f)
fl_predictions = model_fl.predict_proba(temp_f)



##get Features_FL_train
#for i in range(int(len_A_fl/2)):
#    temp=A_fl.iloc[i,0:6].values.astype(float)
#    temp_2=A_fl.iloc[i,8:10].values.astype(float)
#    temp=np.concatenate((temp,temp_2),axis=0)
#    if sum(temp)>1:
#        Features_fl_train.append(temp)
#        labels_fl_train.append(1)
#    
#for i in range(int(len_B_fl/2)):
#    temp=B_fl.iloc[i,1:5].values.astype(float)
#    temp_2=B_fl.iloc[i,8:10].values.astype(float)
#    temp=np.concatenate((temp,temp_2),axis=0)
#    Features_fl_train.append(temp)
#    labels_fl_train.append(2)
#    
#Features_fl_train = np.squeeze(Features_fl_train)
#labels_fl_train=np.squeeze(labels_fl_train)
#
##train model
#model_lensf.fit(Features_lensf_train, labels_lensf_train)
#model_fl.fit(Features_fl_train, labels_fl_train)
#
################ test 
##lensfree subimages for A
#A_lensf_test = []
#for i in range(int(len_A_lensf/2),int(len_A_lensf)):
#    temp=A_lensf.iloc[i,11:156].values.astype(float)
#    A_lensf_test.append(temp)
#A_lensf_test=np.squeeze(A_lensf_test)
#
##fluorescent subimages for A 
#A_fl_test = []
#for i in range(int(len_A_fl/2),int(len_A_fl)):
#    temp=A_fl.iloc[i,1:5].values.astype(float)
#    temp_2=A_fl.iloc[i,8:10].values.astype(float)
#    temp=np.concatenate((temp,temp_2),axis=0)
#    A_fl_test.append(temp)
#A_fl_test=np.squeeze(A_fl_test)
#
##lensfree subimages for B
#B_lensf_test = []
#for i in range(int(len_B_lensf/2),int(len_B_lensf)):
#    temp=B_lensf.iloc[i,11:156].values.astype(float)
#    B_lensf_test.append(temp)
#B_lensf_test=np.squeeze(B_lensf_test)
#
##fluorescent subimages for B 
#B_fl_test = []
#for i in range(int(len_B_fl/2),int(len_B_fl)):
#    temp=B_fl.iloc[i,1:5].values.astype(float)
#    temp_2=B_fl.iloc[i,8:10].values.astype(float)
#    temp=np.concatenate((temp,temp_2),axis=0)
#    B_fl_test.append(temp)
#B_fl_test=np.squeeze(B_fl_test)
#
##test using svm
#A_lensf_predictions = model_lensf.predict_proba(A_lensf_test)
#A_lensf_predictions[:,0]=A_lensf_predictions[:,0]
#A_lensf_predictions[:,1]=A_lensf_predictions[:,1]
#A_lensf_predictions=np.rint(A_lensf_predictions)
#print( 'sum(A_lensf_predictions)'+ str(sum(A_lensf_predictions)))
#sum(A_lensf_predictions)
# 
#B_lensf_predictions = model_lensf.predict_proba(B_lensf_test)
#B_lensf_predictions[:,0]=B_lensf_predictions[:,0]
#B_lensf_predictions[:,1]=B_lensf_predictions[:,1]
#B_lensf_predictions=np.rint(B_lensf_predictions)
#print( 'sum(B_lensf_predictions)'+ str(sum(B_lensf_predictions)))
#sum(B_lensf_predictions)
#
#A_fl_predictions = model_fl.predict_proba(A_fl_test)
#A_fl_predictions=np.rint(A_fl_predictions)
#print( 'sum(A_fl_predictions)')
#sum(A_fl_predictions)
#
#B_fl_predictions = model_fl.predict_proba(B_fl_test)
#B_fl_predictions=np.rint(B_fl_predictions)
#print( 'sum(B_fl_predictions)')
#sum(B_fl_predictions)
#
#####save svm model
#pickle.dump( model_lensf, open( "model_lensf.sav", "wb" ) )
#pickle.dump( model_fl, open( "model_fl.sav", "wb" ) )

