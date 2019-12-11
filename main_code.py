# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:44:28 2019

@author: xiong
"""
from particle_class import particle
import cv2
import pickle
import numpy as np
import pandas as pd
import glob
import os
#import matplotlib.pyplot as plt
import copy
import math
import tkinter as tk
from tkinter import filedialog
#import xlwt
  
def calculateDistance(p1,p2):  
     dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  
     return dist  

def draw_bbox (img, bbox, color):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, p1, p2, (color), 2, 1)
    return img

def bbox2point (bbox):
    position_x=(int (bbox[0] + bbox[2]/2))
    position_y=(int (bbox[1] + bbox[3]/2))
    return position_x, position_y

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
    return box_ps

def point2bbox_image(image,x,y,WIDE_WINDOW,HEIGHT_WINDOW):
    height,wide= np.shape(image)
    image_out=np.zeros((1,1),int)
    if x+WIDE_WINDOW/2<wide and x-WIDE_WINDOW/2>0:
        if y+HEIGHT_WINDOW/2<height and y-HEIGHT_WINDOW/2>0:
            image_out=image[int(y-HEIGHT_WINDOW/2):int(y+HEIGHT_WINDOW/2),int(x-WIDE_WINDOW/2):int(x+WIDE_WINDOW/2)]
    return image_out


def create_new_particle(bbox, img_gray, p_List, frame_num):
    #init a particle object
    particle_a= particle(bbox, img_gray)
    #init tracker
    particle_a.tracker_create() 
    ok = particle_a.tracker.init(img_gray, bbox)
    #save position of object
    particle_a.get_save_position(frame_num, bbox)
    #add this particles into the list
    p_List.append(particle_a)

def check_overlap_bbox(bbox_1 , bbox_2):
    overlap_flag = 0
    if bbox_1[2] > 0 and bbox_2[2] > 0:
        x1 = bbox_1[0] + bbox_1[2]/2
        x2 = bbox_2[0] + bbox_2[2]/2
        y1 = bbox_1[1] + bbox_1[3]/2
        y2 = bbox_2[1] + bbox_2[3]/2
        dem=0# demension of overlap area
        # if overlap area is too much
        #calculate the overlapped dem of two box
        if abs(x1-x2)< (bbox_1[2]/2+bbox_2[2]/2):
            if abs(y1-y2)< (bbox_1[3]/2+bbox_2[3]/2):
                dem= (bbox_1[2]/2+bbox_2[2]/2 - abs(x1-x2))*((bbox_1[3]/2+bbox_2[3]/2)-abs(y1-y2))
        # if area is too big, set flag
        if dem>0.5*bbox_1[2]*bbox_1[3]:
            overlap_flag=1
        if dem>0.5*bbox_2[2]*bbox_2[3]:
            overlap_flag=1
    return overlap_flag

###hog
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
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmInit(C=12.5, gamma=0.50625):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  

    
#reading and configuartion
root = tk.Tk()
root.withdraw()

images_folder = filedialog.askdirectory()
D_THOR=20

CONTRAST_EH=2

#initial 
frame_count=0
frame_num=0
last_frame_num=0
particle_List = []
current_position = np.zeros((2,1),np.float32)
current_prediction = np.zeros((4,1),np.float32)
Distance_1 = 500
Distance_2 = 500
font = cv2.FONT_HERSHEY_SIMPLEX
num_algae=0
num_bead=0
num_detected = 0
found_particles=[]
frame_particles=[]
current_particles=[]
period_n=0
period_flag=0

# initialize pandas dataframe
data = []
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['frame', 'PN', 'speed_x', 'speed_y', 'location_x','location_y', 'lensf', 'lensf_subimage','fl_subimage', 'case','period_n']) 



#creat a excel
#book = xlwt.Workbook(encoding="utf-8")
#sheet1 = book.add_sheet("summary")
#sheet2 = book.add_sheet("note")

#sliding windows config
#compress in the y direction.
RATIO_HEIGHT=1.3
STRIDE_WINDOW=16
WIDE_WINDOW=4*16
HEIGHT_WINDOW=4*16
num_w_wide=int((1640-WIDE_WINDOW)/STRIDE_WINDOW+1)
num_w_height=int((1232/RATIO_HEIGHT-HEIGHT_WINDOW)/STRIDE_WINDOW+1)
windows_A=np.zeros((num_w_height,num_w_wide,HEIGHT_WINDOW,WIDE_WINDOW),dtype=np.uint8)
windows_label=np.zeros((num_w_height+10,num_w_wide+10),dtype=np.uint8)

CostOfNonAssignment=STRIDE_WINDOW*STRIDE_WINDOW*6
CostOfNonPerfect=CostOfNonAssignment/5

#config HOG
hog = get_hog()
hog_descriptors = []

# Now create a new SVM & load the model:
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

    
for f in glob.glob(os.path.join(images_folder, "*.bmp")):
    #check if it is a lensfree image or a fluorescence image
    
    ##########process fluorescence image
    if f[-5]=='f':
        #count period number
        period_flag=1
        
        frame = cv2.imread(f)
        img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#        CONTRAST_EH=3#enhacne contrast of gray images
#        img_gray=cv2.multiply(img_gray-128+int(128/CONTRAST_EH),CONTRAST_EH)#enhacne contrast of gray images
#        
#        img_gray = cv2.GaussianBlur(img_gray,(3,3),0)#blurring
#        
        img_gray_draw= copy.deepcopy(img_gray)
        last_frame_num=copy.deepcopy(frame_num)
        frame_num= f[-10:-5]
        frame_num = int(frame_num)
        print(frame_num)
        f_save=copy.deepcopy(f)
        
        num_particle=len(particle_List)
        
        for j in range(num_particle):
            '''
            update kalman
            update kalman filter according to the frame_num
            save prediction,i.e, speed and predicted position in the class particle
            Don NOT corrent kalman filter. correct kalman filter in the end of this frame processing
            '''
            #when kalman is available 
            if particle_List[j].kalman_ok:
                for i in range(frame_num-last_frame_num):
                    current_prediction = particle_List[j].kalman.predict()
                #save prediction position
                particle_List[j].save_prediction(current_prediction)
                #correct position with prediction
                particle_List[j].save_position(frame_num, particle_List[j].position_prediction)
                
        
        for j in range(num_particle):      
            #get the particle positon in the fluorescent images
            x_fl=int(particle_List[j].position_x[-1])
            y_fl=int(particle_List[j].position_y[-1]*RATIO_HEIGHT + 65)
            
            #kill this particle if it has move out the lensfree video
            if y_fl-65>1232:
                particle_List[j].appear = -10
                print('del:',particle_List[j].appear)
            
            if particle_List[j].PN > 0:
                #update speed
                particle_List[j].speed_update()
                speed_temp=copy.deepcopy(particle_List[j].speed)
                
                #get the sub_image that need to be saved
                sub_image = point2bbox_image(img_gray,x_fl,y_fl, 70, 120)
                
                #write data into pandas dataframe
                ##pd.DataFrame['frame', 'PN', 'speed', 'location_x','location_y', 'lensf', 'lensf_subimage', 'case']) 
                df=df.append({'frame' : frame_num ,'PN' :particle_List[j].PN, 'speed_x':speed_temp[0],'speed_y':speed_temp[1],'location_x':particle_List[j].position_x[-1], 'location_y':particle_List[j].position_y[-1], 'lensf': 0,'fl_subimage': sub_image , 'case': particle_List[j].case, 'period_n':period_n}, ignore_index = True) 
                
                ##write data in excel   
                #sheet1.write(frame_count,2*particle_List[j].PN,particle_List[j].position_x[-1])
                #sheet1.write(frame_count,2*particle_List[j].PN+1,particle_List[j].position_y[-1])
                cv2.circle(img_gray_draw,(x_fl,y_fl), 60, 128, 3)
                cv2.circle(img_gray_draw,(x_fl,y_fl), 35, 128, 3)
                cv2.putText(img_gray_draw, str(particle_List[j].PN), (x_fl,y_fl), font, 1, (128), 1, cv2.LINE_AA)
            else:
                if particle_List[j].kalman_ok==1:
                    cv2.circle(img_gray_draw,(x_fl,y_fl), 20, 128, 1)
        
    ######## process lensfree images
    if not f[-5]=='f':
        #count period number
        if period_flag==1:
            period_n=period_n+1
        period_flag=0
        
        frame = cv2.imread(f)
        img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #correct img in height direction
        img_gray = cv2.resize(img_gray, (1640,int(1232/RATIO_HEIGHT)), interpolation = cv2.INTER_AREA)
        
        #preprocess
        img_gray=cv2.multiply(img_gray-128+int(128/CONTRAST_EH),CONTRAST_EH)#enhacne contrast of gray images
        img_gray = cv2.GaussianBlur(img_gray,(3,3),0)#blurring
        
        img_gray_draw= copy.deepcopy(img_gray)
        last_frame_num=copy.deepcopy(frame_num)
        frame_num= f[-9:-4]
        frame_num = int(frame_num)
        print(frame_num) 
        f_save=copy.deepcopy(f)
        
        #remove dead object. if only one object left,do not delete it.
        point_p=0
        num_particle=len(particle_List)
        for j in range(num_particle):
            if particle_List[point_p].appear <= 0:
                #if only one object left,do not delete it.
                if len(particle_List)>1:
                    del particle_List[point_p]
            else:
                point_p=point_p+1
        
    #    # if 1st frame, create particles with hog, init tracking
    #    if frame_count==1:
    #        #detect particles in img
    #        dets = detector(img_gray)
    #        #init tracker 
    #        for k, d in enumerate(dets):
    #            #get bbox from dets
    #            bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
    #            create_new_particle(bbox, img_gray, particle_List, frame_num)
    #            #label particles with square and save
    #            draw_bbox (img_gray, bbox, 0)
    #            cv2.imwrite(f_save,img_gray) 
    #            
    #        continue
        
    #    '''
    #    if this frame is flourescent images
    #    '''
    #    type_tem='unknown'
    #    # if this frame is not lensfree frame
    #    if frame_num not in lensfree_frames:
    #        num_particle=len(particle_List)
    #        for j in range(num_particle):
    #            '''
    #            Prediction. each particles in a list
    #            '''
    #            if particle_List[j].kalman_ok:
    #                for i in range(frame_num-last_frame_num):
    #                    current_prediction = particle_List[j].kalman.predict()
    #                particle_List[j].save_prediction(current_prediction)
    #                p1 = (int(current_prediction[0]-50), int(current_prediction[1]-50))
    #                p2 = (int(current_prediction[0]+50), int(current_prediction[1]+50))
    #                '''
    #                Collect fluorescent signal
    #                '''
    #                if p1[0]>= 0 and p1[1]>= 0 and p2[0]< 1640 and p2[1]<700:
    #                    img_fl=img_gray[p1[1]:p2[1],p1[0]:p2[0]]
    #                    particle_List[j].save_fl_img(frame_num,img_fl)
    #                    '''
    #                    Check if algae
    #                    ''' 
    #                    if particle_List[j].type == 'unknown':
    #                        type_tem = particle_List[j].check_algae() 
    #                        if type_tem == 'algae':
    #                            num_algae = num_algae+1
    #                        if type_tem == 'PS_bead':
    #                            num_bead = num_bead + 1
    #    
    #        #label particles
    #        for j in range(num_particle):
    #            if particle_List[j].kalman_ok:
    #                p1 = (int(particle_List[j].position_prediction[0]-50), int(particle_List[j].position_prediction[1]-70))
    #                p2 = (int(particle_List[j].position_prediction[0]+50), int(particle_List[j].position_prediction[1]+30))
    #                cv2.rectangle(img_gray, p1, p2, (255), 2, 1)
    #                cv2.putText(img_gray,str(particle_List[j].PN) +':' + particle_List[j].type, p1, font, 1, (255), 1, cv2.LINE_AA)
    #                
    #            #show summary
    #        cv2.putText(img_gray, 'algae num:' + str(num_algae) ,(100,950), font, 3, (0), 3, cv2.LINE_AA)
    #        cv2.putText(img_gray, 'bead num:' + str(num_bead) ,(100,1150), font, 3, (0), 3, cv2.LINE_AA)
    #    
    #        cv2.imwrite(f_save,img_gray) 
    #        continue
        
        
        '''
        detect particles with hog in img
        '''
    
        
        # convert windows into vectors
        descriptor = hog.compute(img_gray,(STRIDE_WINDOW,STRIDE_WINDOW),(0,0))       
        hog_descriptors = np.reshape(descriptor, (-1, int(WIDE_WINDOW*HEIGHT_WINDOW/STRIDE_WINDOW/STRIDE_WINDOW*9)))
        hog_descriptors = np.squeeze(hog_descriptors)
        
        #predict using svm_proba
        predictions = model.predict_proba(hog_descriptors)
        predictions = predictions[:,1]
        windows_prediction = np.reshape(predictions, (num_w_height,num_w_wide))
        windows_prediction=windows_prediction*255
        windows_prediction=windows_prediction.astype(np.uint8)
        #windows_prediction_2=deepcopy(windows_prediction)
        ret,windows_prediction = cv2.threshold(windows_prediction,140,255,cv2.THRESH_BINARY)    
    
        #find overlap hod detection. And reduce overlap section using     
    #    contour_list = []
    #    image, contours, hierarchy = cv2.findContours(windows_prediction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #    for k in range(len(contours)):
    #        box_ps = contours2box_ps(contours[k])
    #        bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
    #        kernel = np.ones((2,2),np.uint8)
    #        if bbox[2]>4 or bbox[3]>4:
    #            windows_prediction[box_ps[1]:box_ps[3]:,box_ps[0]:box_ps[2]]=cv2.erode(windows_prediction[box_ps[1]:box_ps[3]:,box_ps[0]:box_ps[2]],kernel,iterations = 1)
    
    
        # save HOG results to contour_list
        contour_list = []
        contours, hierarchy = cv2.findContours(windows_prediction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for k in range(len(contours)):
            box_ps = contours2box_ps(contours[k])
            bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
            [detection_x,detection_y]=bbox2point(bbox)
            mask_hog = np.zeros(windows_prediction.shape,np.uint8)
            cv2.drawContours(mask_hog,contours[k],-1,255,-1)
            windows_prediction_contour= cv2.bitwise_and(mask_hog,windows_prediction)
        #    print(windows_prediction_contour.sum())
        #    print(detection_x,detection_y)
            if windows_prediction_contour.sum()>320:
                bbox = (int(detection_x*STRIDE_WINDOW),int(detection_y*STRIDE_WINDOW),WIDE_WINDOW,HEIGHT_WINDOW)
                contour_list.append(bbox)
                draw_bbox (img_gray_draw, bbox, 255)
        hog_fake=np.zeros(len(contour_list),int)
                
                
        #update tracking and update kalman prediction
        num_particle=len(particle_List)
        for j in range(num_particle):
            '''
            update tracker 
            save position in the class article
            save bbox in the class article
            '''
            particle_List[j].tracker_update(img_gray,frame_num)#update tracker
            #draw tracking bbox
           # draw_bbox (img_gray, particle_List[j].bbox, 0)
            #cv2.putText(img_gray, str(j),tuple([particle_List[j].position_x[-1],particle_List[j].position_y[-1]]), font, 1, (255), 1, cv2.LINE_AA)
        # remove the bad tracking.     
        # if movement is more than thorshold, set positon to -100,-100,
        for j in range(num_particle):
            for i in range(num_particle-j-1):
                overlap_flag = check_overlap_bbox(particle_List[j].bbox , particle_List[j+i+1].bbox)
                if overlap_flag:
                    particle_List[j].get_save_position(frame_num, (-100,-100,-100,-100))
                    particle_List[j+i+1].get_save_position(frame_num, (-100,-100,-100,-100))
        
            
        for j in range(num_particle):
            '''
            update kalman
            update kalman filter according to the frame_num
            save prediction,i.e, speed and predicted position in the class particle
            Don NOT corrent kalman filter. correct kalman filter in the end of this frame processing
            '''
            #when kalman is available 
            if particle_List[j].kalman_ok:
                for i in range(frame_num-last_frame_num):
                    current_prediction = particle_List[j].kalman.predict()
                #save prediction position
                particle_List[j].save_prediction(current_prediction)
    
                #get distance bwtween predicted position and tracking position
                particle_List[j].distance_k2t=calculateDistance(current_prediction,[particle_List[j].position_x[-1],particle_List[j].position_y[-1]])
                #cv2.circle(img_gray,(current_prediction[0],current_prediction[1]), 25, 255, 1)
                    
        #get cost_matrix, the distance square between tracking and detection
        cost_matrix = np.zeros((len(contour_list),num_particle))+10000
        for j in range(num_particle):
            for i in range(len(contour_list)):
                #get position of each contours
                x_det,y_det= bbox2point(contour_list[i])
                #get position of each particle
                x_track=particle_List[j].position_x[-1]
                y_track=particle_List[j].position_y[-1]
                #save distance in matrix
                cost_matrix[i,j]=(x_det-x_track)**2+(y_det-y_track)**2
        
        
        
        #get cost_matrix using prediction, the distance square between prediction and detection
        cost_matrix_predict = np.zeros((len(contour_list),num_particle))+10000
        for j in range(num_particle):
            if particle_List[j].kalman_ok:
                for i in range(len(contour_list)):
                    #get position of each contours
                    x_det,y_det= bbox2point(contour_list[i])
                    #get position of each particle
                    x_prediction=particle_List[j].position_prediction[0]
                    y_prediction=particle_List[j].position_prediction[1]
                    #save distance in matrix
                    cost_matrix_predict[i,j]=(x_det-x_prediction)**2+(y_det-y_prediction)**2
                    
                    
    #  get indexs for pair with 1 proity 
        if len(contour_list)>0 and num_particle>0:
            hog2track=np.zeros(len(contour_list),int)-1
            hog2predict=np.zeros(len(contour_list),int)-1
            track2hog=np.zeros(num_particle,int)-1
            predict2hog=np.zeros(num_particle,int)-1
    
            for j in range(num_particle):
                minElement = np.amin(cost_matrix[:,j])
                if minElement < CostOfNonAssignment:
                    track2hog[j] = np.where(cost_matrix[:,j] == minElement)[0][0]
                minElement = np.amin(cost_matrix_predict[:,j])
                if minElement < CostOfNonAssignment:
                    predict2hog[j] = np.where(cost_matrix_predict[:,j] == minElement)[0][0] 
                
            for i in range(len(contour_list)):
                minElement = np.amin(cost_matrix[i,:])
                if minElement < CostOfNonAssignment:
                    hog2track[i] = np.where(cost_matrix[i,:] == minElement)[0][0] 
                minElement = np.amin(cost_matrix_predict[i,:])
                if minElement < CostOfNonAssignment:
                    hog2predict[i] = np.where(cost_matrix_predict[i,:] == minElement)[0][0] 
              
            #find overlap hog and label fake this hog detection using prediction
            for i in range(len(contour_list)):
                temp = np.where(predict2hog==i)
                if len(temp[0]) > 1:
                    #release predict pair
                    predict2hog[temp[0]]=-2
                    #report fake hog
                    hog2predict[i]=-2
    
                    hog_fake[i]=1
                    
            #find overlap hog and label fake this hog detection using tracking
            for i in range(len(contour_list)):
                temp = np.where(track2hog==i)
                if len(temp[0]) > 1:
                    #release predict pair
                    track2hog[temp[0]]=-2
                    #report fake hog
                    hog2track[i]=-2
                    hog_fake[i]=1
    
            #to fix the bug that two detections merge to one detection
            #del fake hog and set relavant particle to unknow,disable tracking results
            i_del=0
            for i in range(len(contour_list)):
                if hog_fake[i] > 0:
    #                hog_fake[i] = 0
                    cost_matrix=np.delete(cost_matrix, i-i_del, 0)
                    cost_matrix_predict=np.delete(cost_matrix_predict, i-i_del, 0)
                    del contour_list[i-i_del]
                    hog2predict = np.delete(hog2predict,i-i_del,0)
                    hog2track = np.delete(hog2track,i-i_del,0)
                    i_del=i_del+1
                    
            #disable tracking and set prediction to unknown   
            for j in range(num_particle):
                if predict2hog[j]== -2 or track2hog[j] ==-2:
                    cost_matrix[:,j]=cost_matrix[:,j]*0+10000
                    cost_matrix_predict[:,j]=cost_matrix_predict[:,j]*0+10000
                    #set tracking results to false
                    particle_List[j].get_save_position(frame_num, (-100,-100,-100,-100))
            
    #        #use cost_matrix_predict to find overlapped detection 
    #        for i in range(len(contour_list)):
    #            for j in range(num_particle):
    #                if cost_matrix_predict[i,j] < CostOfNonAssignment:
    #                    hog_fake[i]=hog_fake[i]+1
    #            if hog_fake[i]>1:
    #                cost_matrix[i,:]=cost_matrix[i,:]*0+10000
    #                cost_matrix_predict[i,:]=cost_matrix_predict[i,:]*0+10000
    #            else:
    #                hog_fake[i]=0
            
    
         
        #fix a bug. case when no detection found. add a fake detection result to let np.amin(cost_matrix[x,x]) run
        if len(contour_list)==0:
            cost_matrix = np.zeros((1,num_particle))+10000
            cost_matrix_predict = np.zeros((1,num_particle))+10000
        
        #assignDetectionsToTracks(costMatrix,costOfNonAssignment)
        hog_new=np.zeros(len(contour_list),int)-1
    
        for j in range(num_particle):
            particle_List[j].index_2h=-1
            minElement = np.amin(cost_matrix[:,j])
            hog_i = np.where(cost_matrix[:,j] == minElement)
            hog_i=hog_i[0][0]
            #case assigned tracking 
            if minElement < CostOfNonAssignment: #case 1,2
                #if this particle is not assigned to a det
                if hog_new[hog_i] == -1:
                    if particle_List[j].kalman_ok:
                        particle_List[j].case = 1
                    else:
                        particle_List[j].case = 2  
                    hog_new[hog_i]=j
                    particle_List[j].index_2h= hog_i
                        
            #case unassigned tracking
            else: #case 3,4,5,6
                if particle_List[j].kalman_ok: #case 3,4,6 
                    x_track=particle_List[j].position_x[-1]
                    y_track=particle_List[j].position_y[-1]
                    [x_predict,y_predict]= particle_List[j].position_prediction
                    distance=(x_predict-x_track)**2+(y_predict-y_track)**2
    
                    if distance < CostOfNonAssignment:
                        particle_List[j].case = 4
                    else:
                        minElement = np.amin(cost_matrix_predict[:,j])
                        hog_i = np.where(cost_matrix_predict[:,j] == minElement)
                        hog_i=hog_i[0][0]
                        #########fix the bug that two detection merge to one particles when tracking is missing.
                        if minElement < CostOfNonAssignment:
                            particle_List[j].case = 3
                            hog_new[hog_i]=j
                            particle_List[j].index_2h= hog_i
                        else:
                            particle_List[j].case = 6
                else:
                    particle_List[j].case = 5
                    
         # unassignedDetections= case 3,0
         # case unassigned detecton
    #    for i in range(len(contours)):
    #        if hog_new[i]==-1:              
    #            Distance_k2h = 500
    #            #get current det position
    #            box_ps = contours2box_ps(contours[i])
    #            bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
    #            for j in range(num_particle):
    #                current_position[0],current_position[1]= bbox2point(bbox) 
    #                # find the min distance between det and kalman prediction
    #                #when kalman is available 
    #                if particle_List[j].kalman_ok:
    #                    if calculateDistance(current_position,particle_List[j].position_prediction) < Distance_k2h:
    #                        Distance_k2h=calculateDistance(current_position,particle_List[j].position_prediction) 
    #                        particle_List[j].hog_bbox = copy.deepcopy(bbox)
    #                        particle_List[j].index_k2h = copy.deepcopy(i) 
    #                        hog_new[i]==j
    #            if Distance_k2h < CostOfNonAssignment:#case 3
    #                particle_List[hog_new[i]].case = 3
    #            
    #            else:
    #                hog_new[i]=-1#case 0
                    
         
        '''
        Deal with found particles
        （1）if detected and tracked matched: add points, correct Kalman
        （2）if detected and tracked matched, kalman not initial: add points, init Kalman
        (3)if detected and tracked not matched, Kalman match detection: correct Kalman, correct tracking
        (4)if detected and tracked not matched, Kalman match tracking: correct Kalman, 
        (5)if detected and tracked not matched, kalman not initial: loss points
        (6)if detected， and tracked not matched, kalman not match anything: loss points, correct position with Kalman
        （0）Rest of hug detected particles, create new object
    
        '''
        num_particle=len(particle_List)
        for j in range(num_particle):
            if particle_List[j].case == 1: 
                particle_List[j].report_found()
    #            #correct kalman filter 
                current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
                particle_List[j].kalman.correct(current_position)     
    #           set number detected
                if particle_List[j].appear>5:
                    ok = particle_List[j].assign_PN (num_detected)
                    if ok:
                        num_detected= num_detected + 1
                            
            elif particle_List[j].case == 2:
                #check if unsigned predict closed to unsigned detection
                mindistance=16*10000
                #find min distance between unsigned predict and this detection
                fake_j=-1
                for j_temp in range(num_particle):
                    if predict2hog[j_temp]==-1:
                        if not j_temp==j:                     
                            if particle_List[j_temp].kalman_ok: 
                                x_track=particle_List[j].position_x[-1]
                                y_track=particle_List[j].position_y[-1]
                                [x_predict,y_predict]= particle_List[j_temp].position_prediction
                                distance=(x_predict-x_track)**2+(y_predict-y_track)**2
                                if distance < mindistance:
                                    mindistance = distance
                                    fake_j = j_temp
                                    
                if mindistance < 9*CostOfNonAssignment:
                    ok = particle_List[j].assign_PN (particle_List[fake_j].PN)
                    particle_List[fake_j].appear = 0
                    predict2hog[fake_j]=-2
                    #print(fake_j)
                #init kalman filter
                particle_List[j].init_kalman()
                #add point
                #particle_List[j].report_found()
    
            elif particle_List[j].case == 3: 
                #get bbox of target detection
                hog_i = np.where(hog_new == j)
                if len(hog_i[0]):
                    hog_i=hog_i[0][0]
                    bbox = contour_list[hog_i]
                    #init tracker 
                    particle_List[j].tracker_create() 
                    ok = particle_List[j].tracker.init(img_gray, bbox)
                    particle_List[j].get_save_position(frame_num, bbox)
                    #correct kalman filter
                    current_position[0],current_position[1]= bbox2point(bbox)       
                    particle_List[j].kalman.correct(current_position)
            elif particle_List[j].case == 4:                     
                #correct kalman filter
                current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
                particle_List[j].kalman.correct(current_position)
            elif particle_List[j].case == 5:             
                particle_List[j].appear = 0
            elif particle_List[j].case == 6:    
                particle_List[j].report_missing()
                #disable tracking
                particle_List[j].tracking_ok = 0
                particle_List[j].save_position(frame_num, particle_List[j].position_prediction)
    
      
        #remove dead object. if only one object left,do not delete it.
        point_p=0
        num_particle=len(particle_List)
        for j in range(num_particle):
            if particle_List[point_p].appear <= 0:
                #if only one object left,do not delete it.
                if len(particle_List)>1:
                    del particle_List[point_p]
            else:
                point_p=point_p+1
                
    
        
        #case 0
        i_det=0
        for bbox in contour_list:
            if hog_new[i_det]==-1:
    #            if hog_fake[i_det]==0:
                create_new_particle(bbox, img_gray, particle_List, frame_num)
            i_det = i_det + 1
        
        # label and summary 
        num_particle=len(particle_List)
        print('num of tracking object:' + str(num_particle))
        for j in range(num_particle):        
            if particle_List[j].PN > 0:
                #update speed
                particle_List[j].speed_update()
                speed_temp=copy.deepcopy(particle_List[j].speed)
                #get the sub_image that need to be saved
                sub_image = point2bbox_image(img_gray,particle_List[j].position_x[-1],particle_List[j].position_y[-1],WIDE_WINDOW,HEIGHT_WINDOW)
                
                #write data into pandas dataframe
                ##pd.DataFrame['frame', 'PN', 'speed', 'location_x','location_y', 'lensf', 'lensf_subimage', 'case']) 
                df=df.append({'frame' : frame_num ,'PN' :particle_List[j].PN, 'speed_x':speed_temp[0],'speed_y':speed_temp[1],'location_x':particle_List[j].position_x[-1], 'location_y':particle_List[j].position_y[-1], 'lensf': 1,'lensf_subimage': sub_image , 'case': particle_List[j].case, 'period_n': period_n}, ignore_index = True) 
                
                ##write data in excel   
                #sheet1.write(frame_count,2*particle_List[j].PN,particle_List[j].position_x[-1])
                #sheet1.write(frame_count,2*particle_List[j].PN+1,particle_List[j].position_y[-1])
                cv2.circle(img_gray_draw,(particle_List[j].position_x[-1],particle_List[j].position_y[-1]), 50, 0, 3)
                cv2.putText(img_gray_draw, str(particle_List[j].PN)+':'+str(particle_List[j].case), (particle_List[j].position_x[-1],particle_List[j].position_y[-1]), font, 1, (255), 1, cv2.LINE_AA)
            else:
                if particle_List[j].kalman_ok==1:
                    cv2.circle(img_gray_draw,(particle_List[j].position_x[-1],particle_List[j].position_y[-1]), 40, 0, 1)
                
                
                
    #        if particle_List[j].appear > 0:
    #            cv2.circle(img_gray,tuple(particle_List[j].position_prediction), 50, 0, 1)
    ##            cv2.circle(img_gray,(particle_List[j].position_x[-1],particle_List[j].position_y[-1]), 50, 0, 1)
    #            #draw_bbox (img_gray, particle_List[j].bbox, 255)
    #            #cv2.putText(img_gray, str(j)+':'+str(particle_List[j].case),tuple([particle_List[j].position_x[-1],particle_List[j].position_y[-1]]), font, 1, (255), 1, cv2.LINE_AA)
    #            cv2.putText(img_gray, str(particle_List[j].PN) + ':' + str(particle_List[j].case), tuple(particle_List[j].position_prediction), font, 1, (255), 1, cv2.LINE_AA)
    #            
            
            #
    
        #show summary
        #cv2.putText(img_gray, 'algae num:' + str(num_algae) ,(100,950), font, 3, (0), 3, cv2.LINE_AA)
        #cv2.putText(img_gray, 'bead num:' + str(num_bead) ,(100,1150), font, 3, (0), 3, cv2.LINE_AA)
        
    #    #output img make for semi-superviosed learning
    #    #select active aera
    #    windows_label=np.zeros((num_w_height+10,num_w_wide+10),dtype=np.uint8)
    #    windows_label_2=np.zeros((num_w_height,num_w_wide),dtype=np.uint8)
    #    windows_label_2=windows_label_2.astype(np.uint8)
    #    for i in range(2,100):
    #        for j in range(37,137):
    #            windows_label_2[i,j]=1
    #
    #    #get tracking active area
    #    for j in range(num_particle):
    #        [track_x,track_y]=bbox2point(particle_List[j].track_bbox)    
    #        temp_h=int((track_y-HEIGHT_WINDOW/2)/STRIDE_WINDOW)
    #        temp_w=int((track_x-WIDE_WINDOW/2)/STRIDE_WINDOW)
    #        windows_label[temp_h,temp_w] = windows_label[temp_h,temp_w]+1
    #        windows_label[temp_h+1,temp_w] = windows_label[temp_h+1,temp_w]+1
    #        windows_label[temp_h,temp_w+1] = windows_label[temp_h,temp_w+1]+1
    #        windows_label[temp_h+1,temp_w+1] = windows_label[temp_h+1,temp_w+1]+1
    #
    #    #get prediction active area
    #    for j in range(num_particle):     
    #        temp_h=int((particle_List[j].position_prediction[1]-HEIGHT_WINDOW/2)/STRIDE_WINDOW)
    #        temp_w=int((particle_List[j].position_prediction[0]-WIDE_WINDOW/2)/STRIDE_WINDOW)
    #        windows_label[temp_h,temp_w] = windows_label[temp_h,temp_w]+1
    #        windows_label[temp_h+1,temp_w] = windows_label[temp_h+1,temp_w]+1
    #        windows_label[temp_h,temp_w+1] = windows_label[temp_h,temp_w+1]+1
    #        windows_label[temp_h+1,temp_w+1] = windows_label[temp_h+1,temp_w+1]+1      
    #    
    #    windows_label=windows_label[0:num_w_height,0:num_w_wide]
    #    windows_label=windows_label+3*windows_prediction/255
    #    windows_label=windows_label*51
    #    windows_label=windows_label.astype(np.uint8) 
    #    
    #    windows_label = cv2.bitwise_and(windows_label, windows_label, mask=windows_label_2)
    #    windows_label_2= 150*windows_label_2-150
    #    windows_label = windows_label+windows_label_2
    #    cv2.imwrite(f_save[0:-4]+'label.jpg',windows_label) 
        
    ###save images and summary
    cv2.imwrite(f_save,img_gray_draw) 
    print('algae num:' + str(num_algae))
    print('bead num:' + str(num_bead))
    print('all num:' + str(num_detected))
    frame_particles.append(frame_num)
    found_particles.append(num_detected)
    current_particles.append(num_particle)
    
    if frame_num%500==0:
        pickle.dump( df, open(images_folder + "/results.p", "wb" ) )
        
pickle.dump( df, open(images_folder + "/results.p", "wb" ) )
    