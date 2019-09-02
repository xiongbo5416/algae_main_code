# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:43:20 2019

@author: xiong
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


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
#    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    


root = tk.Tk()
root.withdraw()

STRIDE_WINDOW=8
WIDE_WINDOW=4*16
HEIGHT_WINDOW=6*16
num_w_wide=int((1640-WIDE_WINDOW)/STRIDE_WINDOW+1)
num_w_height=int((1232-HEIGHT_WINDOW)/STRIDE_WINDOW+1)
windows_A=np.zeros((num_w_height,num_w_wide,HEIGHT_WINDOW,WIDE_WINDOW),dtype=np.uint8)
windows_label=np.random.rand(num_w_height+10,num_w_wide+10)
#windows_label=windows_label+0.9
windows_label=windows_label.astype(np.uint8)


fig_path = filedialog.askopenfilename()
img=cv2.imread(fig_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#show image
cv2.namedWindow("please read console", cv2.WINDOW_NORMAL) 
cv2.imshow('please read console',img)
#find out how many particles and non-articles in this image, and press any key
cv2.waitKey(0)
num_particles = input ("Enter number of particles")
num_particles = int(num_particles)
num_unknown = input ("Enter number of non-particles")
num_unknown = int(num_unknown)
cv2.destroyAllWindows()

for i in range(num_w_height):
    for j in range(num_w_wide):
        windows_A[i,j,:,:]=img[i*STRIDE_WINDOW:i*STRIDE_WINDOW+HEIGHT_WINDOW,j*STRIDE_WINDOW:j*STRIDE_WINDOW+WIDE_WINDOW]


#select particles
for i in range(num_particles):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    bbox = cv2.selectROI('output',img, True)
    cv2.destroyAllWindows()
    print("    <box top='" + str(bbox[1]) + "' left ='" + str(bbox[0]) + "' width ='" + str(bbox[2]) + "' height ='" + str(bbox[3]) + "'/>" )
    img_cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    
    #add label
    temp_h=int((bbox[1]+bbox[3]/2-HEIGHT_WINDOW/2)/STRIDE_WINDOW)
    temp_w=int((bbox[0]+bbox[2]/2-WIDE_WINDOW/2)/STRIDE_WINDOW)
    
    #set unknown
    for h_i in range(max([0,temp_h-2]),min([num_w_height,temp_h+4])):
        for w_i in range(max([0,temp_w-1]),min([num_w_wide,temp_w+3])):
            windows_label[h_i,w_i] = 1

    #set 1
    windows_label[temp_h,temp_w] = 2
    windows_label[temp_h+1,temp_w] = 2
    windows_label[temp_h,temp_w+1] = 2
    windows_label[temp_h+1,temp_w+1] = 2
    
                
#    cv2.imshow('1',windows_A[temp_h,temp_w,:,:])
#    cv2.imshow('2',windows_A[temp_h+1,temp_w+1,:,:])
#    cv2.imshow('3',windows_A[temp_h,temp_w-2,:,:])
#    cv2.imshow('4',windows_A[temp_h-4,temp_w,:,:])
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    #cv2.imwrite(fig_path +'_cut.jpg',img_cut) 

#select unknow
for i in range(num_unknown):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    bbox = cv2.selectROI('output',img, True)
    cv2.destroyAllWindows()
    print("    <box top='" + str(bbox[1]) + "' left ='" + str(bbox[0]) + "' width ='" + str(bbox[2]) + "' height ='" + str(bbox[3]) + "'/>" )
    img_cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    
    #add label
    temp_h=int((bbox[1]+bbox[3]/2-HEIGHT_WINDOW/2)/STRIDE_WINDOW)
    temp_w=int((bbox[0]+bbox[2]/2-WIDE_WINDOW/2)/STRIDE_WINDOW)
    
    #set unknown
    for h_i in range(max([0,temp_h-2]),min([num_w_height,temp_h+4])):
        for w_i in range(max([0,temp_w-1]),min([num_w_wide,temp_w+3])):
            windows_label[h_i,w_i] = 1

cv2.imwrite(fig_path[0:-4] +'label.jpg',windows_label*127) 
