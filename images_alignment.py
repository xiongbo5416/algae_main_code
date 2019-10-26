# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:43:29 2019

@author: xiong
"""

import numpy as np
import cv2
#import matplotlib.pyplot as plt
import tkinter
from tkinter.filedialog import askopenfilename

#how many points used in alighment ?
NUM_POINTS=4
root = tkinter.Tk()
root.withdraw()
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing


#select lensfree image
lensfree_name = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(lensfree_name)
img_1 = cv2.imread(lensfree_name)
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)



#select fl imaging
fl_name = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(fl_name)
img_2 = cv2.imread(fl_name)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.multiply(gray_2,1)


#point1s are in the lensfree images
cv2.namedWindow("please read console", cv2.WINDOW_NORMAL) 
points1 = np.zeros((NUM_POINTS, 2), dtype=np.int32)
for i in range(NUM_POINTS):
    #select points. left top conner shoould be points
    bbox = cv2.selectROI('please read console',img_1, False)
    points1[i]=(bbox[0],bbox[1])
    #print points
    print("points1[" + str(i) + "]=(" + str(points1[i][0]) + "," + str(points1[i][1]) + ")")

cv2.destroyAllWindows()
#points1[0]=(1240,180)
#points1[1]=(1135,437)
#points1[2]=(902,633)
#points1[3]=(732,679)
#points1[4]=(848,947)
#points1[5]=(329,578)
#points1[6]=(280,333)
#points1[7]=(1501,989)

cv2.namedWindow("please read console", cv2.WINDOW_NORMAL) 
points2 = np.zeros((NUM_POINTS, 2), dtype=np.int32)
for i in range(NUM_POINTS):
    #select points. left top conner shoould be points
    bbox = cv2.selectROI('please read console',gray_2, False)
    points2[i]=(bbox[0],bbox[1])
    #print points
    print("points2[" + str(i) + "]=(" + str(points2[i][0]) + "," + str(points2[i][1]) + ")")

cv2.destroyAllWindows()
#points2 = np.zeros((4, 2), dtype=np.float32)

#points1[0]=(435,20)
#points1[1]=(843,218)
#points1[2]=(959,834)
#points1[3]=(208,791)
#points2[0]=(1534,650)
#points2[1]=(1382,343)
#points2[2]=(905,243)
#points2[3]=(928,798)

#
    
#
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
# Use homography
height, width = gray_1.shape
gray_2_reg = cv2.warpPerspective(gray_2, h, (width, height))


cv2.namedWindow("1", cv2.WINDOW_NORMAL) 
cv2.imshow('1',gray_2_reg)
cv2.namedWindow("2", cv2.WINDOW_NORMAL) 
cv2.imshow('2',gray_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#imgplot = plt.imshow(gray_2_reg-gray_1)
#plt.show()