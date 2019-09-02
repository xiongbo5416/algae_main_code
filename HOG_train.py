# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:43:20 2019

@author: xiong
"""
import cv2
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
import glob
import os
import copy

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
    
def svmInit(C=5, gamma=0.50625):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  
  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()


root = tk.Tk()
root.withdraw()

STRIDE_WINDOW=16
WIDE_WINDOW=4*16
HEIGHT_WINDOW=6*16
#num_w_wide=int((1640-WIDE_WINDOW)/STRIDE_WINDOW+1)
#num_w_height=int((1232-HEIGHT_WINDOW)/STRIDE_WINDOW+1)
#windows_A=np.zeros((num_w_height,num_w_wide,HEIGHT_WINDOW,WIDE_WINDOW),dtype=np.uint8)
#windows_label=np.zeros((num_w_height+10,num_w_wide+10),dtype=np.uint8)


#config HOG
hog = get_hog()

model = svmInit()
hog_descriptors = []
labels_train=[]

for fig_i in range(1):
    fig_path = filedialog.askopenfilename()
    img=cv2.imread(fig_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    
    fig_path_2 = fig_path[0:-4]+'label.jpg'
    img_label=cv2.imread(fig_path_2)
    windows_label=cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
    windows_label=windows_label/100
    windows_label=windows_label.astype(int)
    
    hog_descriptors = []
    train_label_0 = []
    train_label_1 = []
    location_0 = list()
    location_1 = list()
    
    height=len(img)
    width=len(img[0,:])
    for i in range (0,width-WIDE_WINDOW):
        for j in range (0,height-HEIGHT_WINDOW):
            #print(i)
            if windows_label[j,i]==0:
                location_0.append((i,j))
                train_label_0.append(0)
            if windows_label[j,i]==2:
                location_1.append((i,j))
                train_label_1.append(1)          

#descriptor = hog.compute(img,(16,16),(0,0),location_0)                  
    
#cv2.namedWindow("output", cv2.WINDOW_NORMAL)
#cv2.imshow('output',windows_A[56,30,:,:])
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#hog_descriptors = np.squeeze(hog_descriptors)
#labels_train = np.squeeze(labels_train)
#svmTrain(model, hog_descriptors, labels_train)
#model.save("svm_model.xml")

#predictions = svmPredict(model, hog_descriptors)