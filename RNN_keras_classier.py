# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:42:02 2020

@author: xiong
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def get_RNN_feature_label(A,ST,END,LABEL):
    max_PN=A['PN'].max(axis = 0, skipna = True)
    features_out = []
    labels_out=[]
    for i in range(int(max_PN*ST)+1,int(max_PN*END)+1):
        print(i)
        PN_data=A.loc[A['PN']==i]
        ##fix the bug that PN particles has no data
        if len(PN_data)==0:
            continue
        PN_data=PN_data.sort_values(by=['frame'])
        F_LENGTH=20
        F_SIZE=3
        features= np.zeros((1,F_LENGTH,F_SIZE))
        temp_data=PN_data['lensf'].values.astype(float)
        temp_len=len(temp_data)
        temp_data=np.append(temp_data,PN_data['predict_A'].values.astype(float))
        temp_data=np.append(temp_data,PN_data['predict_B'].values.astype(float))
        temp_data=np.reshape(temp_data,(-1,temp_len))
        temp_data=temp_data.T
        
        if len(temp_data)>F_LENGTH:
            features[0,:,:]=temp_data[0:F_LENGTH,:]
        else:
            features[0,0:len(temp_data),:]=temp_data
        if len(features_out) == 0:
            features_out = features
            labels_out = np.append(labels_out,LABEL)
        else:
            features_out = np.append(features_out,features,axis=0)
            labels_out = np.append(labels_out,LABEL)
      
    return features_out, labels_out

#####random.shuffle
def features_labels_shuffle(train_x,train_y):
    shape_1,shape_2,shape_3=train_x.shape
    #temp arrary
    data=np.zeros((shape_1,shape_2+1,shape_3))
    data[:,0:shape_2,:]=train_x
    for i in range(shape_3):
        data[:,shape_2,i]=train_y
    #shuffle
    np.random.shuffle(data)
    
    #get return
    train_x_out=data[:,0:shape_2,:]
    train_y_out=data[:,shape_2,0]
    
    return train_x_out,train_y_out




print(tf.__version__)

root = tk.Tk()
root.withdraw()

#select dataframe files
print('select green algae')
dpname = filedialog.askopenfilename()
print(dpname)
A = pickle.load( open( dpname, "rb" ) )

#select dataframe files
print('select lensfree related to particle 10um beads')
dpname = filedialog.askopenfilename()
print(dpname)
B = pickle.load( open( dpname, "rb" ) )

len_A=len(A)
len_B=len(B)


######## generate train dataset
train_x_1, train_y_1 = get_RNN_feature_label(A,0,0.7,1)
train_x_2, train_y_2 = get_RNN_feature_label(B,0,0.7,0)

train_x=np.append(train_x_1,train_x_2,axis=0)
train_y=np.append(train_y_1,train_y_2,axis=0)

del train_x_1, train_y_1 
del train_x_2, train_y_2 


#####random.shuffle
train_x,train_y=features_labels_shuffle(train_x,train_y)

######## generate test dataset
test_x_1, test_y_1 = get_RNN_feature_label(A,0.7,1,1)
test_x_2, test_y_2 = get_RNN_feature_label(B,0.7,1,0)

test_x=np.append(test_x_1,test_x_2,axis=0)
test_y=np.append(test_y_1,test_y_2,axis=0)

del test_x_1, test_y_1 
del test_x_2, test_y_2 


            
model = tf.keras.Sequential()


# Add a LSTM layer with 28 internal units.
model.add(layers.LSTM(5, input_shape=(20, 3)))

# Add a Dense layer with 10 units and softmax activation.
model.add(layers.Dense(2, activation='softmax'))

model.summary()



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train_images = tf.reshape(train_images,[-1,28])


model.fit(train_x, train_y,
          validation_data=(test_x,  test_y),
          batch_size=10,
          epochs=8)

#model.fit(train_images, train_labels, epochs=10)
#
#
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#print('\nTest accuracy:', test_acc)

predictions = model.predict(test_x)