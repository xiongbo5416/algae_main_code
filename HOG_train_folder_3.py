# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:43:20 2019

@author: xiong
"""
import cv2
#import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
import glob
import os
from sklearn.svm import SVC

#import copy
import pickle

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
    #affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
        
def svmInit(C=5, gamma=0.50625):
    clf = SVC(gamma='auto')
    clf.probability=True
    clf.kernel='rbf'
    #clf.kernel='poly'
    clf.C=5
  
    return clf

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def contours2box_ps(contours):
    x2=0
    y2=0
    x1=32000
    y1=32000
    length=len(contours)
    for i in range(length):
        if contours[i][0][0]>x2:
            x2=contours[i][0][0]
        if contours[i][0][0]<x1:
            x1=contours[i][0][0]
        if contours[i][0][1]>y2:
            y2=contours[i][0][1]
        if contours[i][0][1]<y1:
            y1=contours[i][0][1]
    box_ps=[x1,y1,x2,y2]
    #print(box_ps)
    return box_ps

def bbox2point (bbox):
    position_x=(int (bbox[0] + bbox[2]/2))
    position_y=(int (bbox[1] + bbox[3]/2))
    return position_x, position_y



root = tk.Tk()
root.withdraw()

STRIDE_WINDOW=16
WIDE_WINDOW=4*16
HEIGHT_WINDOW=4*16
#compress in the y direction.
RATIO_HEIGHT=1.3
num_w_wide=int((1640-WIDE_WINDOW)/STRIDE_WINDOW+1)
num_w_height=int((1232/RATIO_HEIGHT-HEIGHT_WINDOW)/STRIDE_WINDOW+1)
windows_A=np.zeros((num_w_height,num_w_wide,HEIGHT_WINDOW,WIDE_WINDOW),dtype=np.uint8)


#config HOG
hog = get_hog()

model = svmInit()
hog_descriptors = []
labels_train=[]



images_folder = filedialog.askdirectory()
for f in glob.glob(os.path.join(images_folder, "*.bmp")):
    ##if it is raw image
    if f[-8:-4] != 'Copy':
        img=cv2.imread(f)
        
        #contrast enhance
        CONTRAST_EH=2
        img=cv2.multiply(img-128+int(128/CONTRAST_EH),CONTRAST_EH)
        #correct img in height direction
        img = cv2.resize(img, (1640,int(1232/RATIO_HEIGHT)), interpolation = cv2.INTER_AREA)
        print(f[:-4] )
        
        '''
        get hog descriptor
        '''
        descriptor = hog.compute(img,(STRIDE_WINDOW,STRIDE_WINDOW),(0,0))       
        #hog_descriptors = np.reshape(descriptor, (-1, 864))
        descriptor =np.reshape(descriptor,(num_w_height,num_w_wide,int(WIDE_WINDOW*HEIGHT_WINDOW/STRIDE_WINDOW/STRIDE_WINDOW*9)))
        descriptor_temp=descriptor[:,:,-1]
        #save label and its position
        mask_window_yes=np.zeros((num_w_height+10,num_w_wide+10),np.uint8)
        mask_window_uncertain=np.zeros((num_w_height+10,num_w_wide+10),np.uint8)
        mask_window_no=np.zeros((num_w_height+10,num_w_wide+10),np.uint8)
        mask_window_random=np.random.rand(num_w_height+10,num_w_wide+10)
        mask_window_random=mask_window_random+0.5 #(1-x) windows will be in the training sample
        mask_window_random=mask_window_random.astype(np.uint8)
        
        #read label imag
        f_2 = f[0:-4]+' - Copy.bmp'
        img_label=cv2.imread(f_2)
        img_label = cv2.resize(img_label, (1640,int(1232/RATIO_HEIGHT)), interpolation = cv2.INTER_AREA)
        img_label_r=img_label[:,:,2]
        img_label_g=img_label[:,:,1]
        img_label_b=img_label[:,:,0]
        '''
        get red dot which refer to particles label
        '''
        #get mask
        ret,thresh1 = cv2.threshold(img_label_r,200,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(img_label_g,50,255,cv2.THRESH_BINARY_INV)
        mask_yes=cv2.bitwise_and(thresh1, thresh1, mask=thresh2)
        #using contours to get location of center point and save its location
        location_yes=[]
        contours, hierarchy = cv2.findContours(mask_yes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for k in range(len(contours)):
            #find center position of each contour
            box_ps = contours2box_ps(contours[k])
            bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
            [detection_x,detection_y]=bbox2point(bbox)
            #use point position to find location of windows
            temp_h=int((detection_y-HEIGHT_WINDOW/2)/STRIDE_WINDOW)
            temp_w=int((detection_x-WIDE_WINDOW/2)/STRIDE_WINDOW)
            #save point position
            location_yes.append((temp_w,temp_h))
            #change window label
            mask_window_yes[temp_h,temp_w]=1
            
        '''
        get white dot which refer to uncertain label
        '''
        ret,thresh1 = cv2.threshold(img_label_b,210,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(img_label_g,210,255,cv2.THRESH_BINARY)
        mask_uncertain=cv2.bitwise_and(thresh1, thresh1, mask=thresh2)
        location_uncertain=[]
        contours, hierarchy = cv2.findContours(mask_uncertain,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for k in range(len(contours)):
            #find center position of each contour
            box_ps = contours2box_ps(contours[k])
            bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
            [detection_x,detection_y]=bbox2point(bbox)
            #use point position to find location of windows
            temp_h=int((detection_y-HEIGHT_WINDOW/2)/STRIDE_WINDOW)
            temp_w=int((detection_x-WIDE_WINDOW/2)/STRIDE_WINDOW)
            #save point position
            location_uncertain.append((temp_w,temp_h))
            mask_window_uncertain[temp_h,temp_w]=1
            
        #change window label
        kernel=np.ones((4,4))
        mask_window_uncertain = cv2.dilate(mask_window_uncertain,kernel,iterations = 1)  
        mask_window_uncertain_2 = cv2.dilate(mask_window_yes,kernel,iterations = 1) 
        mask_window_uncertain = cv2.bitwise_or(mask_window_uncertain,mask_window_uncertain_2)
        
        kernel=np.ones((8,8))
        mask_window_no = cv2.dilate(mask_window_yes,kernel,iterations = 1)
        mask_window_no = cv2.bitwise_xor(mask_window_uncertain,mask_window_no)

        kernel=np.ones((2,2))
        mask_window_yes = cv2.dilate(mask_window_yes,kernel,iterations = 1)
        
        
        
        mask_window_final = cv2.bitwise_or(mask_window_yes,mask_window_random)
        mask_window_final = cv2.bitwise_and((1-mask_window_no),mask_window_final)
        mask_window_final = cv2.bitwise_or(mask_window_uncertain,mask_window_final)

        mask_window_final = mask_window_final+mask_window_yes
        
        #save svm training sample
        for i in range(num_w_wide):
            for j in range(num_w_height):
                if mask_window_final[j,i]==2:
                    hog_descriptors.append(descriptor[j,i,:])
                    labels_train.append(1)
                if mask_window_final[j,i]==0:
                    if (descriptor[j,i,0]+descriptor[j,i,-1])>0:
                        hog_descriptors.append(descriptor[j,i,:])
                        labels_train.append(0)
    else:
        continue
    
#    for i in range(num_w_height):
#        for j in range(num_w_wide):
#            windows_A[i,j,:,:]=img[i*STRIDE_WINDOW:i*STRIDE_WINDOW+HEIGHT_WINDOW,j*STRIDE_WINDOW:j*STRIDE_WINDOW+WIDE_WINDOW]
#    
#    for i in range(num_w_height):
#        for j in range(num_w_wide):
#            if windows_label[i,j]==0:
#                descriptor = hog.compute(windows_A[i,j,:,:])
#                if sum(descriptor)>0:
#                    hog_descriptors.append(descriptor)
#                    labels_train.append(0)
#            if windows_label[i,j]==2:
#                descriptor = hog.compute(windows_A[i,j,:,:])
#                hog_descriptors.append(descriptor)
#                labels_train.append(1)
#                
#    
hog_descriptors = np.squeeze(hog_descriptors)
labels_train = np.squeeze(labels_train)


cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow('output',mask_window_final.astype(np.uint8)*100)
cv2.waitKey(0)
cv2.destroyAllWindows()

model.fit(hog_descriptors, labels_train)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


