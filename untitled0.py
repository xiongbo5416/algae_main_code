# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:32:44 2020

@author: xiong
"""

import pickle

import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()

dpname = filedialog.askopenfilename()
print(dpname)
A = pickle.load( open( dpname, "rb" ) )