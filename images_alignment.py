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
gray_2 = cv2.multiply(gray_2,3)
#flip horizental
#gray_2 = cv2.flip(gray_2, 1)
gray_2 = gray_2.T
#flip horizental
#gray_2 = cv2.flip(gray_2, 1)


#point1s are in the lensfree images
cv2.namedWindow("please read console", cv2.WINDOW_NORMAL) 
points1 = np.zeros((NUM_POINTS, 2), dtype=np.float32)
for i in range(NUM_POINTS):
    #select points. left top conner shoould be points
    bbox = cv2.selectROI('please read console',img_1, True)
    points1[i]=(bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2)
    #print points
    print("points1[" + str(i) + "]=(" + str(points1[i][0]) + "," + str(points1[i][1]) + ")")

cv2.destroyAllWindows()



cv2.namedWindow("please read console", cv2.WINDOW_NORMAL) 
points2 = np.zeros((NUM_POINTS, 2), dtype=np.float32)
for i in range(NUM_POINTS):
    #select points. left top conner shoould be points
    bbox = cv2.selectROI('please read console',gray_2, True)
    points2[i]=(bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2)
    #print points
    print("points2[" + str(i) + "]=(" + str(points2[i][0]) + "," + str(points2[i][1]) + ")")

cv2.destroyAllWindows()
#points2 = np.zeros((4, 2), dtype=np.float32)

#points1[0]=(1120,1066)
#points1[1]=(912,873)
#points1[2]=(645,226)
#points1[3]=(940,96)
#points2[0]=(684,177)
#points2[1]=(842,378)
#points2[2]=(1288,552)
#points2[3]=(1370,342)
#
    
#
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
# Use homography

#h = cv2.getAffineTransform(points2, points1)
#use 3 point

height, width = gray_1.shape
#gray_2_reg = cv2.warpAffine(gray_2, h, (width, height))
gray_2_reg = cv2.warpPerspective(gray_2, h, (width, height))
cv2.imwrite(lensfree_name,gray_2_reg)

cv2.namedWindow("1", cv2.WINDOW_NORMAL) 
cv2.imshow('1',gray_2_reg)
cv2.namedWindow("2", cv2.WINDOW_NORMAL) 
cv2.imshow('2',gray_1)
cv2.waitKey(0)
cv2.destroyAllWindows()



#imgplot = plt.imshow(gray_2_reg-gray_1)
#plt.show()