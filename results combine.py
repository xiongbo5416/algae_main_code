# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:17:34 2020

@author: xiong
"""
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import cv2


root = tk.Tk()
root.withdraw()


print('select a .p file')
dpname = filedialog.askopenfilename()
result_1 =pickle.load( open( dpname, "rb" ) )
result_1 ['frame'].describe()
result_1 ['PN'].describe()

print('select a .p file')
dpname = filedialog.askopenfilename()
result_2 =pickle.load( open( dpname, "rb" ) )
result_2 ['frame'].describe()
result_2 ['PN'].describe() 

result_2=result_2[result_2['frame']>38146]


#result_2 ['frame'] = result_2 ['frame'] + 26961
#result_2 ['PN']= result_2 ['PN']+ 3235-11
#
#pickle.dump( result, open("results.p", "wb" )  )

#result=pd.concat([result_1,result_2])
