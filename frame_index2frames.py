# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:08:36 2019

@author: xiong
"""


import pickle
import numpy as np


PERIOD=14# 14 OR 15
LENSFR_P=6#6 OR 7
FL_P=5#5 OR 6

results = pickle.load( open( "frames_index.p", "rb" ) )
shape=np.shape(results)

for i in range(2,shape[0]-1):
    if results[i][0]==0:
        break
    if results[i+1][0]-results[i][0] == PERIOD or results[i+1][0]-results[i][0] == PERIOD+1:
        pass
    else:
        print(results[i][0])

