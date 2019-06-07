# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:44:28 2019

@author: xiong
"""
from particle_class import particle
import cv2
import dlib
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import copy
import math

  
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
    
#reading and configuartion
PATH='C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/0514/'
FOLDER_name= '6_5'
detector = dlib.simple_object_detector("detector.svm")
D_THOR=20

#initial 
frame_count=0
frame_num=0
last_frame_num=0
particle_List = []
current_position = np.zeros((2,1),np.float32)
current_prediction = np.zeros((4,1),np.float32)
Distance_1 = 500
Distance_2 = 500



    
for f in glob.glob(os.path.join(PATH + FOLDER_name, "*.jpg")):
    frame = cv2.imread(f)
    img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_frame_num=copy.deepcopy(frame_num)
    frame_num= f[-8:-4]
    frame_num = int(frame_num)
    print(frame_num) 
    frame_count=frame_count+1
    f_save=copy.deepcopy(f)
    
    # if 1st frame, create particles with hog, init tracking
    if frame_count==1:
        #detect particles in img
        dets = detector(img_gray)
        #init tracker 
        for k, d in enumerate(dets):
            #get bbox from dets
            bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
            create_new_particle(bbox, img_gray, particle_List, frame_num)
            #label particles with square and save
            draw_bbox (img_gray, bbox, 0)
            cv2.imwrite(f_save,img_gray) 
            
        continue

    # after 2nd frame
    '''
    detect particles with hog in img
    '''
    dets = detector(img_gray)
    #draw hog detection with black square
    for k, d in enumerate(dets):
        #get bbox from dets
        bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
        draw_bbox (img_gray, bbox, 0)
            
            
    #update tracking and update kalman prediction
    num_particle=len(particle_List)
    for j in range(num_particle):
        '''
        update tracker 
        save position in the class article
        save bbox in the class article
        '''
        ok, bbox = particle_List[j].tracker.update(img_gray)#update tracker
        particle_List[j].get_save_position(frame_num, bbox)
        
        
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
        
        
        '''
        update distance
        between tracking and hog
        between kalman predict and hog
        '''
        Distance_t2h = 500
        Distance_k2h = 500
        i_det=-1
        for k, d in enumerate(dets):
            #get current position
            i_det=i_det+1
            bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
            current_position[0],current_position[1]= bbox2point(bbox)
           
            # find the min distance between hog and tracker
            if calculateDistance(current_position,[particle_List[j].position_x[-1],particle_List[j].position_y[-1]]) < Distance_t2h:
                Distance_t2h=calculateDistance(current_position,[particle_List[j].position_x[-1],particle_List[j].position_y[-1]])
                particle_List[j].index_t2h = copy.deepcopy(i_det)
                
            # find the min distance between hog and kalman prediction
            #when kalman is available 
            if particle_List[j].kalman_ok:
                if calculateDistance(current_position,particle_List[j].position_prediction) < Distance_k2h:
                    Distance_k2h=calculateDistance(current_position,particle_List[j].position_prediction) 
                    particle_List[j].hog_bbox = copy.deepcopy(bbox)
                    particle_List[j].index_k2h = copy.deepcopy(i_det)
        
        #save min distance in the class
        particle_List[j].distance_t2h = copy.deepcopy(Distance_t2h)
        particle_List[j].distance_k2h = copy.deepcopy(Distance_k2h)
    
    # deal with particels
    num_particle=len(particle_List)
    for j in range(num_particle):
        if particle_List[j].distance_t2h < D_THOR: 
            if particle_List[j].kalman_ok: #case 1
                #add point
                particle_List[j].report_found()
                #correct kalman filter
                current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
                particle_List[j].kalman.correct(current_position)
            else: #case 2
                #add point
                particle_List[j].report_found()
                #init kalman filter
                particle_List[j].init_kalman()

        else:
            if particle_List[j].distance_k2h < D_THOR: #case 3
                #init tracker
                particle_List[j].tracker_create() 
                ok = particle_List[j].tracker.init(img_gray, particle_List[j].hog_bbox)
                particle_List[j].get_save_position(frame_num, particle_List[j].hog_bbox)
                #correct kalman filter
                current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
                particle_List[j].kalman.correct(current_position)
                
            elif particle_List[j].distance_k2t < D_THOR: #case 4
                #correct kalman filter
                current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
                particle_List[j].kalman.correct(current_position)
                
            elif not particle_List[j].kalman_ok: #case 5
                particle_List[j].appear = 0

            
            else:#case 6
                particle_List[j].report_missing()
            
                
                
#        '''
#        Case one
#        hog detection, tracking and kalman prediction are matched
#        '''
#        if Distance_1 < D_THOR and Distance_2 < D_THOR:
#           particle_List[j_particle].report_found()
##           # delete current dets
##           del dets[i_dets]
#        
#        '''
#        Case two
#        hog detection, kalman prediction are matched. But tracking does not matched
#        '''
#        if Distance_1 < D_THOR and Distance_2 >= D_THOR:
#            #init tracker
#            particle_List[j_particle].tracker_create() 
#            ok = particle_List[j_particle].tracker.init(img_gray, bbox)
#            
#            #save position, and correct bbox in the class particle
#            particle_List[j_particle].get_save_position(frame_num, bbox)

  
    #remove dead object
    point_p=0
    num_particle=len(particle_List)
    for j in range(num_particle):
        if particle_List[point_p].appear == 0:
            del particle_List[point_p]
        else:
            point_p=point_p+1
        
    # label and summary 
    print(frame_count)
    font = cv2.FONT_HERSHEY_SIMPLEX
    num_particle=len(particle_List)
    for j in range(num_particle):
        if particle_List[j].appear > 0:
            cv2.circle(img_gray,tuple(particle_List[j].position_prediction), 25, 0, 1)
            draw_bbox (img_gray, particle_List[j].bbox, 255)
            cv2.putText(img_gray, str(j),tuple([particle_List[j].position_x[-1],particle_List[j].position_y[-1]]), font, 1, (255), 3, cv2.LINE_AA)
            #print(str(j) + '_t2h:' + str(particle_List[j].distance_t2h))
            #print(str(j) + '_k2t:' + str(particle_List[j].distance_k2t))

        
      
    cv2.imwrite(f_save,img_gray) 
    
    
    

#print(dir(particle_a))