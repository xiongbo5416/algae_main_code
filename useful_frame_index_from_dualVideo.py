# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:51:13 2019

@author: xiong
"""
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import tkinter
from tkinter import filedialog
import pickle


# provide folder path and video name
#PATH='C:/Bo at McMaster/algae data/0514/'
#FOLDER_name= '2_lensfree/'
#FOLDER_name_2='2_fl/'
#NAME_lensfree='2_lensfr.h264'
#NAME_scattering='2_fl.h264'



#select a picture to check whether excitation light is on in the scattering/fl images
root = tkinter.Tk()
root.withdraw()
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

#############################select folder and video
print('select a lensfree video')
videoname = filedialog.askopenfilename()
print(videoname)

print('select a flourescent video')
print('if donnot process fluorescent video, click cancel')
videoname2 = filedialog.askopenfilename()
print(videoname2)


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
N_MOVING=40
bgd=np.zeros((ROL_NUM,COL_NUM,N_MOVING))
bgd_name=[]
i_MOVING=0

#bgd=bgd.astype(float)
threshold_ex = 20 #set it to 0 when want to decide it 
current_frame=0
period_num=0
current_frame_fl=0
# excitation_st=0, excitation light is off, red light is on
# excitation_st=1, excitation light just on, red light just off
# excitation_st>2, excitation light must be on for whole frame
# excitation_st is the num of the frame. The red light has been off before the number of frame
excitation_st=0  
RedOrNot=0
n_moving=0
CONTRAST_EH=3
FL_EH=6
NUM_COMP=1 #since lensfree images and fluorescent images are not start at same time. A compenstion is required.
#creat a list to record the frame No. that is the first frame in each lasers toggle period
MAX_PERIOD=10000 #this define the quatity of red laser burst


list_period = np.zeros(MAX_PERIOD,dtype=np.int32)#record the start frame No in each cycle
lensfr_period = np.zeros(MAX_PERIOD,dtype=np.int32)#record finish cycle in each cycle

fl_st = np.zeros(MAX_PERIOD,dtype=np.int32) #record the start frame No in each cycle
fl_period = np.zeros(MAX_PERIOD,dtype=np.int32)#record finish cycle in each cycle


#when fl video is selected, read one img and select a area 
if len(videoname2) >0:
    #read onet frame to select high light area when threshold_ex=0
    #open a frame when excitation light is on

    print("select a FL frame when RED excitation is on, this frame shall be used to distinguish frames with/without red excitation")
    filename = filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filename + ':', end =" ")
    img = cv2.imread(filename)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_0=img_grey.copy()    
    gray_1=img_grey.copy()
    #img_grey =np.array(img[:, :, 1])
    
    cv2.namedWindow("please select area", cv2.WINDOW_NORMAL) 
    bbox = cv2.selectROI("please select area",img_grey, False)
    cv2.destroyAllWindows()
    
    img_cut = img_grey[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    height, width = img_cut.shape
    #threshold_ex = img_cut.sum()/(height * width)
    #threshold_ex=threshold_ex/4
    print(bbox)
    
    

##################################deal with lens free images
cap=cv2.VideoCapture(videoname)
while period_num<MAX_PERIOD:
    ret, frame = cap.read()
    if frame is None:
        break  
        print('lensfree video done')
    
    
    #print(str(current_frame) + ',' , end =" ")
    
    #frame = cv2.blur(frame,(3,3))
    #create a small frame for display
    frame_s=cv2.resize(frame,(500,500))
    gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
    
    #obtained blue channel of the present frame 
    blue = np.array(frame[:, :, 0])
    red = np.array(frame[:, :, 2])
    green = np.array(frame[:, :, 1])
    #blue_float=blue.astype(float)
    
    #obtain gray images with green channel
    #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = green
    #grey = cv2.fastNlMeansDenoising(grey,5 ,7,21)
#    show images
    cv2.imshow('grey',gray_s)
    
    #blue_profile=sum(blue_float.T)
    #blue_profile=blue_profile/COL_NUM

    
    grey_float=grey.astype(float) 
   
    grey_profile=sum(grey_float.T)
    grey_profile=grey_profile/COL_NUM
    
    print(current_frame)
    #print(max(grey_profile))
    
    #red laser is off
    if  max(grey_profile)>250 or min(grey_profile)<20:
        RedOrNot=0
    
    # red laser just turn on. the first frame in this cycle
    elif RedOrNot == 0:
        RedOrNot=1
        #save bgd frames for moving average
        #temp= cv2.blur(grey,(3,3)) 
        #bgd = np.append(bgd, np.atleast_3d(grey), axis=2)
        #bgd = np.delete(bgd, 0, axis=2)
        #bgd[:,:,i_MOVING] = grey
        #bgd_name.append(current_frame)
        
#        #get bgd subtracted frame
#        bgd_frame=np.mean(bgd,axis=2)
#        gray_dif = grey_float - bgd_frame +128#get difference between presented and previous frames
#        gray_dif=cv2.multiply(gray_dif-128+int(128/CONTRAST_EH),CONTRAST_EH)#enhacne contrast of gray images
#        #save frames
#        if len(foldername)>0:
#            name= foldername +'/' +str(current_frame+1000) +'.jpg'
#            cv2.imwrite(name,gray_dif)

        #record num of start frame per period
        list_period[period_num]=current_frame+10000
        period_num=period_num+1
        lensfr_period[period_num-1]=current_frame+10000
        
        #print(str(current_frame) +':')
        
         #count num in this bgd batch
        #i_MOVING=i_MOVING+1
        
    # red laser has turned on. after second frame in this cycle
    else:   
#        gray_dif = gray_0 - gray_1 +128#get difference between presented and previous frames
#        gray_1=gray_0#renew background frame
#        gray_dif=cv2.multiply(gray_dif-128+int(128/CONTRAST_EH),CONTRAST_EH)#enhacne contrast of gray images
        
        
        #save bgd frames for moving average
        #temp= cv2.blur(grey,(3,3)) 
        #bgd[:,:,i_MOVING] = grey
        #bgd_name.append(current_frame)
        
        #count num in this bgd batch
        #i_MOVING=i_MOVING+1
        #record num of frames in this period
        lensfr_period[period_num-1]=current_frame+10000
        
        
    current_frame += 1

          
    #press "Q" on the keyboard to quit
    if cv2. waitKey(1)& 0xFF == ord('q'):
        break    
cap.release()
cv2.destroyAllWindows()


##################################deal with fouresent images
#for stable particles removel
#fgbg = cv2.createBackgroundSubtractorMOG2()
period_num=0

#when fl video is selected, process fl video
if len(videoname2) >0:    
    #process video
    cap=cv2.VideoCapture(videoname2)
    while current_frame_fl<current_frame:
        #this frame is gray_0, last frame is gray_1.always save save last frame because it is hard to detect when excitation light on/off.
        ret, frame = cap.read()
        if frame is None:
            break 
    
        #create a small frame for display
        #frame_s=cv2.resize(frame,(500,500))
        #gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('frame_2',gray_s)
        gray_0 = np.array(frame[:, :, 2])
        fl_float=gray_0.astype(float) 
        img_cut = gray_0[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        img_cut=img_cut.astype(float) 
        fl_profile=sum(img_cut.T)
        fl_profile=fl_profile/bbox[2]
        #print(max(fl_profile))
        
        
        #blue light is off
        if max(fl_profile)< threshold_ex:
            excitation_st=0
            
        else: #blue light is just on 
           excitation_st = excitation_st+1  
           if excitation_st==1:
               period_num=period_num+1
           if excitation_st==2:#blue light is on for hole frame
               print(str(current_frame_fl))
               fl_st[period_num-1]=current_frame_fl+10000
               fl_period[period_num-1]=current_frame_fl+10000
               print(max(fl_profile))
               
          #when blue light is on
           if excitation_st>2:
                #record finish frame of each cycle
               fl_period[period_num-1]=current_frame_fl-1+10000
#               print(str(current_frame_fl))
#               print(max(fl_profile))

        current_frame_fl=current_frame_fl+1
        '''
        only used for experiment 0910
        '''           


        '''
        #detection using lensfree images

        if len(img_cut)>2 :
            gray_highlight_sum= gray_0[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]].sum()/(height * width)
            print(str(current_frame_fl)+":"+ str(gray_highlight_sum) + ',', end =" ")
        
        # if larger than threshold, the red light is on, excitation light is off
        if gray_highlight_sum > threshold_ex:
           excitation_st=0
           # when previous frame is performed, excitation light is on
           gray_1=gray_0*0#gray_1 is image in the previous frame
        #red light is on in the last frame,but off in this frame
        elif excitation_st==0:
           excitation_st=1
           if current_frame_fl>2:
               period_num=period_num+1
           gray_1=gray_0
           print(str(current_frame_fl)+":"+ str(gray_highlight_sum))
        #excitation light must be on for whole frame
        else:
           excitation_st = excitation_st+1
           if excitation_st>2:

               #alighn scatteing/fl images to lensfree images
               gray_reg= alighment(COL_NUM, ROL_NUM, gray_1)
               
               #gaussian filter to reduce noise
               gray_reg = cv2.GaussianBlur(gray_reg,(9,9),0)
               gray_reg= cv2.multiply(gray_reg,FL_EH)
               #apply background subtraction
               fgmask = fgbg.apply(gray_reg)
               #close precusure to remove noise black pixel inside the particles 
               kernel = np.ones((3,3),np.uint8)
               fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
               #close precusure to remove noise of white pixel outside the particles 
               fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
               #erode and dilate to remove the small particles
               kernel = np.ones((5,5),np.uint8)
               fgmask = cv2.erode(fgmask,kernel,iterations = 1)
               fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
               #apply mask to filter out stable particles
               fg_gray_reg = cv2.bitwise_and(gray_reg,gray_reg, mask= fgmask)
    
    #           #use red square to label the fluorescent particles
    #           lower = np.array([40])
    #           upper = np.array([255])
    #           shapeMask = cv2.inRange(fg_gray_reg, lower, upper)
    #           cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #           edges=cnts[1]
    #           for points in edges:
    #                cv2.rectangle(fg_gray_reg,(points[0][0][0]-30,points[0][0][1]-30),(points[0][0][0]+30,points[0][0][1]+30),(255),3)
    #           #use red square to label the fluorescent particles */
    #
    #            #write note
    #           font = cv2.FONT_HERSHEY_SIMPLEX
    #           cv2.putText(fg_gray_reg,'blue laser on',(50,150), font, 4, (255), 4, cv2.LINE_AA)
               
                #save frames
               if len(foldername)>0:
                   name= foldername +'/' + str(list_period[period_num-1]+excitation_st*2-3+NUM_COMP) +'f.jpg'
                   cv2.imwrite(name,fg_gray_reg) 
                   
               #record num of frames in this period
               fl_period[period_num-1]=list_period[period_num-1]+excitation_st*2-3+NUM_COMP
               #cv2.imwrite(name,cv2.multiply(gray_1,4)) 
               #print(gray_highlight_sum)
           gray_1=gray_0#save this image of this frame
           
        current_frame_fl=current_frame_fl+1
        '''

cap.release()
cv2.destroyAllWindows()
results_output=np.array([list_period, lensfr_period, fl_st, fl_period])
results_output=results_output.T

pickle.dump( results_output, open( "frames_index.p", "wb" ) )



#bgd = np.delete(bgd, 1, axis=2)
#bgd_float=np.mean(bgd,axis=2)
#
#bgd=np.append(bgd,np.atleast_3d(bgd_frame),axis=2)
#a=np.zeros(15)