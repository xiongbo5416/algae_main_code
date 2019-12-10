# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:43:20 2019

@author: xiong
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pickle
from copy import deepcopy

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
    position_x=(bbox[0] + bbox[2]/2)
    position_y=(bbox[1] + bbox[3]/2)
    return position_x, position_y

def draw_bbox (img, bbox, color, thickness):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, p1, p2, (color), thickness , 1)
    return img
  
root = tk.Tk()
root.withdraw()

STRIDE_WINDOW=16
WIDE_WINDOW=4*16
HEIGHT_WINDOW=6*16
num_w_wide=int((1640-WIDE_WINDOW)/STRIDE_WINDOW+1)
num_w_height=int((1232-HEIGHT_WINDOW)/STRIDE_WINDOW+1)
windows_A=np.zeros((num_w_height,num_w_wide,HEIGHT_WINDOW,WIDE_WINDOW),dtype=np.uint8)
windows_label=np.zeros((num_w_height+10,num_w_wide+10),dtype=np.uint8)



fig_path = filedialog.askopenfilename()
img=cv2.imread(fig_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(3,3),0)
#img_2 = img[int(STRIDE_WINDOW/2):,:]
#img_3 = img[:,STRIDE_WINDOW/2:]
#img_4 = img[STRIDE_WINDOW/2:,STRIDE_WINDOW/2:]

    
#config HOG
hog = get_hog()
hog_descriptors = []
# Now create a new SVM & load the model:
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

descriptor = hog.compute(img,(STRIDE_WINDOW,STRIDE_WINDOW),(0,0))       
hog_descriptors = np.reshape(descriptor, (-1, int(WIDE_WINDOW*HEIGHT_WINDOW/STRIDE_WINDOW/STRIDE_WINDOW*9)))
#descriptor =np.reshape(descriptor,(num_w_height,num_w_wide,int(WIDE_WINDOW*HEIGHT_WINDOW/STRIDE_WINDOW/STRIDE_WINDOW*9)))
#descriptor_temp=descriptor[:,:,-1]


#
#                
hog_descriptors = np.squeeze(hog_descriptors)

predictions = model.predict_proba(hog_descriptors)
predictions = predictions[:,1]
windows_prediction = np.reshape(predictions, (num_w_height,num_w_wide))
windows_prediction=windows_prediction*255
windows_prediction=windows_prediction.astype(np.uint8)
windows_prediction_2=deepcopy(windows_prediction)
ret,windows_prediction = cv2.threshold(windows_prediction,140,255,cv2.THRESH_BINARY)
#windows_prediction = cv2.resize(windows_prediction, (num_w_wide*2,num_w_height*2), interpolation = cv2.INTER_AREA)

##for img_2
#descriptor = hog.compute(img_2,(STRIDE_WINDOW,STRIDE_WINDOW),(0,0))       
#hog_descriptors = np.reshape(descriptor, (-1, int(WIDE_WINDOW*HEIGHT_WINDOW/STRIDE_WINDOW/STRIDE_WINDOW*9)))
#hog_descriptors = np.squeeze(hog_descriptors)
#predictions = svmPredict(model, hog_descriptors)
#windows_prediction_2 = np.reshape(predictions, (num_w_height-1,num_w_wide))
#
##for img_3
#descriptor = hog.compute(img_3,(STRIDE_WINDOW,STRIDE_WINDOW),(0,0))       
#hog_descriptors = np.reshape(descriptor, (-1, int(WIDE_WINDOW*HEIGHT_WINDOW/STRIDE_WINDOW/STRIDE_WINDOW*9)))
#hog_descriptors = np.squeeze(hog_descriptors)
#predictions = svmPredict(model, hog_descriptors)
#windows_prediction_3 = np.reshape(predictions, (num_w_height,num_w_wide))
#windows_prediction=windows_prediction.astype(np.uint8)
#
#
##for img_4
#descriptor = hog.compute(img_4,(STRIDE_WINDOW,STRIDE_WINDOW),(0,0))       
#hog_descriptors = np.reshape(descriptor, (-1, int(WIDE_WINDOW*HEIGHT_WINDOW/STRIDE_WINDOW/STRIDE_WINDOW*9)))
#hog_descriptors = np.squeeze(hog_descriptors)
#predictions = svmPredict(model, hog_descriptors)
#windows_prediction_4 = np.reshape(predictions, (num_w_height-1,num_w_wide))
#
#
#
#for i in range(num_w_height-1):
#    for j in range(num_w_wide):
#        windows_prediction[2*i+1,2*j]=windows_prediction_2[i,j]
#        windows_prediction[2*i,2*j+1]=windows_prediction_3[i,j]
#        windows_prediction[2*i+1,2*j+1]=windows_prediction_4[i,j]
#
#kernel = np.ones((2,2),np.uint8)
#windows_prediction  = cv2.dilate(windows_prediction ,kernel,iterations = 1)
#kernel = np.ones((2,2),np.uint8)
#windows_prediction = cv2.erode(windows_prediction,kernel,iterations = 1)


contour_list = []
image, contours, hierarchy = cv2.findContours(windows_prediction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for k in range(len(contours)):
    box_ps = contours2box_ps(contours[k])
    bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
    kernel = np.ones((2,2),np.uint8)
    if bbox[2]>4 or bbox[3]>4:
        windows_prediction[box_ps[1]:box_ps[3]:,box_ps[0]:box_ps[2]]=cv2.erode(windows_prediction[box_ps[1]:box_ps[3]:,box_ps[0]:box_ps[2]],kernel,iterations = 1)

    
contour_list = []
image, contours, hierarchy = cv2.findContours(windows_prediction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for k in range(len(contours)):
    box_ps = contours2box_ps(contours[k])
    bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
    [detection_x,detection_y]=bbox2point(bbox)
    bbox = (int(detection_x*STRIDE_WINDOW),int(detection_y*STRIDE_WINDOW),WIDE_WINDOW,HEIGHT_WINDOW)
    contour_list.append(bbox)
    draw_bbox (img, bbox, (255,0,0),2)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow('output',windows_prediction_2)
#cv2.imshow('output',hog_descriptors)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


for bbox in contour_list:
    print(bbox)