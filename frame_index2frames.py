# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:08:36 2019

@author: xiong
"""


import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import cv2



#purpose: alighment scattering/fl images to lensfree images
#point1 and points2 should be found mannual.point1s are from lensfree images
def alighment(width, height,gray):
    ##flip horizental
    #gray = cv2.flip(gray, 1)
    gray = gray.T
    ##flip horizental
    #gray = cv2.flip(gray, 1)
    points1 = np.zeros((4, 2), dtype=np.float32)
    points2 = np.zeros((4, 2), dtype=np.float32)
#    points1[0]=(1155.5+com_x,1116.0)
#    points1[1]=(342.0+com_x,929.5)
#    points1[2]=(1346.5+com_x,125.0)
#    points1[3]=(263.5+com_x,51.0)
#    points2[0]=(841.0,919.0)
#    points2[1]=(251.0,783.5)
#    points2[2]=(978.5,204.5)
#    points2[3]=(189.0,149.0)
    points1[0]=(717.5,608.0)
    points1[1]=(501.0,366.0)
    points1[2]=(983.0,308.5)
    points1[3]=(504.0,227.0)
    points2[0]=(423.0,524.5)
    points2[1]=(264.0,330.5)
    points2[2]=(626.5,306.5)
    points2[3]=(265.5,225.5)
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    gray_reg = cv2.warpPerspective(gray, h, (width, height))
    #gray_reg =gray
    return gray_reg



root = tk.Tk()
root.withdraw()

PERIOD=14# 14 OR 15
LENSFR_P=6#6 OR 7
FL_P=5#5 OR 6



#############################select file and videos
#where to save these index
print('select a .p file')
print('if donnot have .p file, select the name of the .p file that will be saved in this code')
dpname = filedialog.askopenfilename()

#select videos
print('select a lensfree video')
print('if donnot process lensfree video, click cancel')
videoname = filedialog.askopenfilename()
print(videoname)

print('select a flourescent video')
print('if donnot process fluorescent video, click cancel')
videoname2 = filedialog.askopenfilename()
print(videoname2)

#where to save images
if len(videoname) or len(videoname2)>0:
    print('select a folder to save frame')
    print('if donnot save frames, click cancel')
    foldername = filedialog.askdirectory()
    print(foldername)


#########when the dataframe that saving index has been processed
if dpname[-2:] == '.p':
    ###read the saved dataframe that contain processed index
    df_output =pickle.load( open( dpname, "rb" ) )
    
######## when only lensfree video is going to be processed 
elif len(videoname2)==0:
    ##read lensfree data
    results = pickle.load( open( "frames_index.p", "rb" ) )
    df = pd.DataFrame(results, columns = ['lensf_st', 'lensf_end']) 
    df_2 = pd.DataFrame(results, columns = ['lensf_st_out', 'lensf_end_out']) 
    df_output = pd.concat([df, df_2], axis=1)
    
else:#select a video name when no dpfile saved
    results = pickle.load( open( "frames_index.p", "rb" ) )
    
    ########df contain all information
    data = []
    df = pd.DataFrame(results, columns = ['lensf_st', 'lensf_end', 'fl_st', 'fl_end']) 
    df_2 = pd.DataFrame(data, columns = ['lensf_st_diff','fl_st_diff']) 
    
    df = df.replace(0, np.nan)
    df =df.dropna(axis=0, how='all')
    
    df_2['lensf_st_diff']=df['lensf_st'].diff()
    df_2['fl_st_diff']=df['fl_st'].diff()
    df_2 =df_2.dropna(axis=0, how='all')
    df_2= df_2.set_index(df_2.index-1)
    
    #df_2.iloc[ 0:2 , : ]
    
    df = pd.concat([df, df_2], axis=1)
    del df_2
    df['lensf_length']=df['lensf_end']-df['lensf_st']
    df['fl_length']=df['fl_end']-df['fl_st']
    #df =df.dropna(axis=0, how='any')
    
    ################modify error mannually to make two video alignment
#    df.loc[1820, 'lensf_st_diff'] = 14
    #df.loc[849, 'fl_st_diff'] = 100
#    df.loc[67, 'fl_st_diff'] = 100
#    df.loc[119, 'fl_st_diff'] = 100
#    df.loc[2517, 'lensf_st_diff'] = 100
#    df.loc[982, 'fl_st_diff'] = 100
#    df.loc[1034, 'fl_st_diff'] = 100
#    df.loc[1088, 'fl_st_diff'] = 100
#    df.loc[1140, 'fl_st_diff'] = 100
#    df.loc[79, 'fl_st_diff'] = 100

    
    ########df_error contain error information
    df_error_fl=df.loc[~df['fl_st_diff'].isin([PERIOD,PERIOD+1,np.NaN])]
    df_error_lensf=df.loc[~df['lensf_st_diff'].isin([PERIOD,PERIOD+1,np.NaN])]
    


    
    ##########df_output contain the real frame number that should be save
    ##########correct lensf output framge
    data = []
    df_output_lensf = pd.DataFrame(data, columns = ['lensf_st', 'lensf_end','lensf_st_diff','lensf_length','lensf_st_out', 'lensf_end_out'])
    df_output_lensf['lensf_st']= df['lensf_st']
    df_output_lensf['lensf_st_out']= df['lensf_st']
    df_output_lensf['lensf_st_diff']= df['lensf_st_diff']
    df_output_lensf['lensf_length']= df['lensf_length']
    df_output_lensf['lensf_end']= df['lensf_end']
    df_output_lensf['lensf_end_out']= df['lensf_end']
    df_output_lensf =df_output_lensf.dropna(axis=0, how='any')
    
    (total_rows,total_column)=df_output_lensf.shape
    
    #insert row when a cycle is lost
    index_error= df_error_lensf[df_error_lensf['lensf_st_diff']>PERIOD+1].index.values.astype(int)
    for i in range(len(index_error)):
        line = pd.DataFrame(data, index=[index_error[i]+0.5])
        df_output_lensf = df_output_lensf.append(line, ignore_index=False)
    #sort dataframe by index
    df_output_lensf = df_output_lensf.sort_index().reset_index(drop=True)
    df_output_lensf =df_output_lensf.dropna(axis=0, how='any')
    
    #correct the st and end frame of 
    #when a cycle is lost
    index_error= df_output_lensf[df_output_lensf['lensf_st_diff']>PERIOD+1].index.values.astype(int)
    for i in range(len(index_error)):
       df_output_lensf['lensf_end'][index_error[i]]=np.NaN
       Comp=df_output_lensf['lensf_st_diff'][index_error[i]-1]*2-df_output_lensf['lensf_st_diff'][index_error[i]]
       df_output_lensf.loc[index_error[i]+2:,'lensf_st_out']=df_output_lensf.loc[index_error[i]+2:,'lensf_st_out']+Comp
       df_output_lensf.loc[index_error[i]+2:,'lensf_end_out']=df_output_lensf.loc[index_error[i]+2:,'lensf_end_out']+Comp
       
    #when a cycle is not lost
    index_error= df_output_lensf[df_output_lensf['lensf_st_diff']<PERIOD].index.values.astype(int) #get index when frames are lost
    for i in range(len(index_error)):
       #df_output_lensf['lensf_end'][index_error[i]]=np.NaN 
       Comp=df_output_lensf['lensf_st_diff'][index_error[i]-1]*1-df_output_lensf['lensf_st_diff'][index_error[i]] #compensation 
       df_output_lensf.loc[index_error[i]+1:,'lensf_st_out']=df_output_lensf.loc[index_error[i]+1:,'lensf_st_out']+Comp #compensate after a certain index
       df_output_lensf.loc[index_error[i]+1:,'lensf_end_out']=df_output_lensf.loc[index_error[i]+1:,'lensf_end_out']+Comp#compensate after a certain index
    
    #df_output_lensf =df_output_lensf.dropna(axis=0, how='any')
    
    
    ##########correct fl output framge
    data = []
    df_output_fl = pd.DataFrame(data, columns = ['fl_st', 'fl_end','fl_st_diff','fl_length','fl_st_out', 'fl_end_out'])
    df_output_fl['fl_st']= df['fl_st']
    df_output_fl['fl_st_out']= df['fl_st']
    df_output_fl['fl_st_diff']= df['fl_st_diff']
    df_output_fl['fl_length']= df['fl_length']
    df_output_fl['fl_end']= df['fl_end']
    df_output_fl['fl_end_out']= df['fl_end']
    df_output_fl =df_output_fl.dropna(axis=0, how='any')
    
    (total_rows,total_column)=df_output_fl.shape
    
    #insert row when a cycle is lost
    index_error= df_error_fl[df_error_fl['fl_st_diff']>PERIOD+1].index.values.astype(int)
    for i in range(len(index_error)):
        line = pd.DataFrame(data, index=[index_error[i]+0.5])
        df_output_fl = df_output_fl.append(line, ignore_index=False)
    #sort dataframe by index
    df_output_fl = df_output_fl.sort_index().reset_index(drop=True)
    df_output_fl =df_output_fl.dropna(axis=0, how='any')
    
    ##label the cycles when there is a problem
    #when a cycle is lost
#    index_error= df_output_fl[df_output_fl['fl_st_diff']>PERIOD+1].index.values.astype(int)
#    for i in range(len(index_error)):
#       df_output_fl['fl_end'][index_error[i]]=np.NaN
#    
#    index_error= df_output_fl[df_output_fl['fl_length']<FL_P-1].index.values.astype(int)
#    for i in range(len(index_error)):
#       df_output_fl['fl_end'][index_error[i]]=np.NaN
    
    #when a cycle is not lost
    #index_error= df_output_lensf[df_output_lensf['lensf_st_diff']<PERIOD].index.values.astype(int) #get index when frames are lost
    #for i in range(len(index_error)):
    #   df_output_lensf['lensf_end'][index_error[i]]=np.NaN 
       
       
      
    df_output = pd.concat([df_output_lensf, df_output_fl], axis=1)
    del df_output_lensf
    del df_output_fl
    
    #correct fl_st_out fl_end_out with lensf imformation
    df_output['fl_st_out']= df_output['lensf_end_out']+1
    df_output['fl_end_out']= df_output['fl_st_out']+df_output['fl_length']
    
    print('FL:', df_error_fl.index.values)
    print('Lensf:',df_error_lensf.index.values)
    
    pickle.dump( df_output, open(dpname[0:-5]+".p", "wb" ) )
    

###################process lensfree images
if len(videoname) >0 and len(foldername)>0:
    ####################read first frame of video to get size of video
    cap=cv2.VideoCapture(videoname)
    ret, frame = cap.read()
    gray_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    [ROL_NUM,COL_NUM]=gray_0.shape
    #create an arrary for bgd
    bgd_frame=gray_0.astype(float)
    bgd_frame=bgd_frame*0
    
    
    #######################parameter init
    #Moving average used to got bgd. N_MOVING is a how many frames used in moving average.
    CONTRAST_EH=1
    N_MOVING=40
    bgd=np.zeros((ROL_NUM,COL_NUM,N_MOVING)) #save N_MOVING frames
    bgd_name=[] #save the frame number of N_MOVING frames
    i_MOVING=0
    period_num=0
    #MAX_PERIOD = max(df_output.index.values.astype(int))
    All_PERIOD = df_output.index.values
#    All_PERIOD = np.append(All_PERIOD[1950:2100],All_PERIOD[3275:3360])
#    All_PERIOD = np.append(All_PERIOD[1270:1505],All_PERIOD[1810:1855])
    All_PERIOD = All_PERIOD[0:10]
#    All_PERIOD = All_PERIOD[:188]
    current_frame=10000
    
    
    ####################process lensfree video
    cap=cv2.VideoCapture(videoname)
    for period_num in All_PERIOD:        
        if not df_output.iloc[period_num].isnull().values.any():
            #print(int(df_output.iloc[period_num]['lensf_st_out']),end=" ")
            ##skip the use less images
            while current_frame < df_output.iloc[period_num]['lensf_st']:
                ret, frame = cap.read()
#                #for a preview during video processing
#                frame_s=cv2.resize(frame,(500,500))
#                gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
#                cv2.imshow('preview',gray_s)
                print(current_frame,end='')
                current_frame=current_frame+1
            
            ### process useful images
            frame_compensate= df_output.iloc[period_num]['lensf_st_out']-df_output.iloc[period_num]['lensf_st']
            frame_compensate=int(frame_compensate)
            while current_frame < df_output.iloc[period_num]['lensf_end']+1:
                ret, frame = cap.read()
                #for a preview during video processing
#                frame_s=cv2.resize(frame,(500,500))
#                gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
#                cv2.imshow('preview',gray_s)
                
                #
                gray =np.array(frame[:, :, 1]) #use green channel     
                
                bgd[:,:,i_MOVING] = gray
                bgd_name.append(current_frame+frame_compensate)
                #count num in this bgd batch
                i_MOVING=i_MOVING+1
                
                
                ###################save  batch of frames
                if len(bgd_name) == N_MOVING:        
                    #get bgd subtracted frame
                    bgd_frame=np.mean(bgd,axis=2)
                    #save frames
                    if len(foldername)>0:
                        for i in range(N_MOVING):
                            gray_float=bgd[:,:,i].astype(float) 
                            gray_dif = gray_float - bgd_frame +128#get difference between presented and previous frames
                            gray_dif=cv2.multiply(gray_dif-128+int(128/CONTRAST_EH),CONTRAST_EH)#enhacne contrast of gray images
                            name = foldername +'/' +str(bgd_name[i]) +'.bmp'
                            cv2.imwrite(name,gray_dif)
                    #reset
                    bgd_name=[]
                    i_MOVING=0
                
                print(current_frame)
                current_frame=current_frame+1
                
        #press "Q" on the keyboard to quit
        if cv2. waitKey(1)& 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()
    print('lenfree video done')
    
    
###################process fluorescence images
if len(videoname2) >0 and len(foldername)>0:
    ##init
    current_frame=10000
    CONTRAST_EH_FL=1
    ####################process fluorescence video
    cap=cv2.VideoCapture(videoname2)
    for period_num in All_PERIOD:
        if not df_output.iloc[period_num].isnull().values.any():
            print(int(df_output.iloc[period_num]['fl_st_out']),end=" ")
            ##skip the useless images
            while current_frame < df_output.iloc[period_num]['fl_st']:
                ret, frame = cap.read()
        
                current_frame=current_frame+1
            
            ### process useful images
            frame_compensate= df_output.iloc[period_num]['fl_st_out']-df_output.iloc[period_num]['fl_st']
            frame_compensate=int(frame_compensate)
            
            while current_frame < df_output.iloc[period_num]['fl_end']+1:     
                ret, frame = cap.read()
                gray_fl=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                ###translate to size of lensfree images
                gray_reg= alighment(COL_NUM, ROL_NUM+300, gray_fl)
                
                ####for a preview during video processing
#                frame_s=cv2.resize(frame,(500,500))
#                gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
#                cv2.imshow('preview',gray_s)
                
                #save  frames
                name = foldername +'/' +str(current_frame+ frame_compensate) +'f.bmp'
                gray_reg=cv2.multiply(gray_reg,CONTRAST_EH_FL)#enhacne contrast of gray images
                cv2.imwrite(name,gray_reg)
                current_frame=current_frame+1
                
        #press "Q" on the keyboard to quit
        if cv2. waitKey(0)& 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()
    

