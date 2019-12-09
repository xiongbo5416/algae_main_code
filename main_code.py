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
#import matplotlib.pyplot as plt
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
    
#reading and configuartion
PATH='C:/Users/xiong/OneDrive - McMaster University/Data and files/algae_project/0514/'
FOLDER_name= '6_5'
detector = dlib.simple_object_detector("detector.svm")
D_THOR=20


lensfree_st=[1209,1233,1257,1281,1306,1330,1354,1378,1402,1426,1451,1475,1499]
lensfree_frames = []
for i in range(12):
    lensfree_frames.append(lensfree_st + i*np.ones(13))
lensfree_frames=np.array(lensfree_frames)
lensfree_frames = lensfree_frames.ravel()


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


<<<<<<< Updated upstream
=======
#sliding windows config
STRIDE_WINDOW=16
WIDE_WINDOW=4*16
HEIGHT_WINDOW=6*16
num_w_wide=int((1640-WIDE_WINDOW)/STRIDE_WINDOW+1)
num_w_height=int((1232-HEIGHT_WINDOW)/STRIDE_WINDOW+1)
windows_A=np.zeros((num_w_height,num_w_wide,HEIGHT_WINDOW,WIDE_WINDOW),dtype=np.uint8)
windows_label=np.zeros((num_w_height+10,num_w_wide+10),dtype=np.uint8)

CostOfNonAssignment=STRIDE_WINDOW*STRIDE_WINDOW*6
CostOfNonPerfect=CostOfNonAssignment/4

#config HOG
hog = get_hog()
hog_descriptors = []

# Now create a new SVM & load the model:
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
>>>>>>> Stashed changes

    
for f in glob.glob(os.path.join(PATH + FOLDER_name, "*.jpg")):
    frame = cv2.imread(f)
    img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_frame_num=copy.deepcopy(frame_num)
    frame_num= f[-8:-4]
    frame_num = int(frame_num)
    print(frame_num) 
    frame_count=frame_count+1
    f_save=copy.deepcopy(f)
    
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
    
    '''
    if this frame is flourescent images
    '''
    type_tem='unknown'
    # if this frame is not lensfree frame
    if frame_num not in lensfree_frames:
        num_particle=len(particle_List)
        for j in range(num_particle):
            '''
            Prediction. each particles in a list
            '''
            if particle_List[j].kalman_ok:
                for i in range(frame_num-last_frame_num):
                    current_prediction = particle_List[j].kalman.predict()
                particle_List[j].save_prediction(current_prediction)
<<<<<<< Updated upstream
                p1 = (int(current_prediction[0]-50), int(current_prediction[1]-50))
                p2 = (int(current_prediction[0]+50), int(current_prediction[1]+50))
                '''
                Collect fluorescent signal
                '''
                if p1[0]>= 0 and p1[1]>= 0 and p2[0]< 1640 and p2[1]<700:
                    img_fl=img_gray[p1[1]:p2[1],p1[0]:p2[0]]
                    particle_List[j].save_fl_img(frame_num,img_fl)
                    '''
                    Check if algae
                    ''' 
                    if particle_List[j].type == 'unknown':
                        type_tem = particle_List[j].check_algae() 
                        if type_tem == 'algae':
                            num_algae = num_algae+1
                        if type_tem == 'PS_bead':
                            num_bead = num_bead + 1
=======
                #correct position with prediction
                particle_List[j].save_position(frame_num, particle_List[j].position_prediction)
                
        
        for j in range(num_particle):        
            if particle_List[j].PN > 0:
                #update speed
                particle_List[j].speed_update()
                speed_temp=copy.deepcopy(particle_List[j].speed)
                #get the sub_image that need to be saved
                sub_image = point2bbox_image(img_gray,particle_List[j].position_x[-1],particle_List[j].position_y[-1] + 50, 56, 100)
                
                #write data into pandas dataframe
                ##pd.DataFrame['frame', 'PN', 'speed', 'location_x','location_y', 'lensf', 'lensf_subimage', 'case']) 
                df=df.append({'frame' : frame_num ,'PN' :particle_List[j].PN, 'speed':speed_temp,'location_x':particle_List[j].position_x[-1], 'location_y':particle_List[j].position_y[-1], 'lensf': 0,'fl_subimage': sub_image , 'case': particle_List[j].case, 'period_n':period_n}, ignore_index = True) 
                
                ##write data in excel   
                #sheet1.write(frame_count,2*particle_List[j].PN,particle_List[j].position_x[-1])
                #sheet1.write(frame_count,2*particle_List[j].PN+1,particle_List[j].position_y[-1])
                cv2.circle(img_gray_draw,(particle_List[j].position_x[-1],particle_List[j].position_y[-1]+50), 28, 128, 3)
                cv2.putText(img_gray_draw, str(particle_List[j].PN), (particle_List[j].position_x[-1],particle_List[j].position_y[-1]), font, 1, (128), 1, cv2.LINE_AA)
            else:
                if particle_List[j].kalman_ok==1:
                    cv2.circle(img_gray_draw,(particle_List[j].position_x[-1],particle_List[j].position_y[-1]), 40, 128, 1)
        
    ######## process lensfree images
    if not f[-5]=='f':
        #count period number
        if period_flag==1:
            period_n=period_n+1
        period_flag=0
        
        frame = cv2.imread(f)
        img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #preprocess
        img_gray=cv2.multiply(img_gray-128+int(128/3),3)#enhacne contrast of gray images
        img_gray = cv2.GaussianBlur(img_gray,(3,3),0)#blurring
        
        img_gray_draw= copy.deepcopy(img_gray)
        last_frame_num=copy.deepcopy(frame_num)
        frame_num= f[-9:-4]
        frame_num = int(frame_num)
        print(frame_num) 
        f_save=copy.deepcopy(f)
        

        
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
>>>>>>> Stashed changes
    
        #label particles
        for j in range(num_particle):
            if particle_List[j].kalman_ok:
                p1 = (int(particle_List[j].position_prediction[0]-50), int(particle_List[j].position_prediction[1]-70))
                p2 = (int(particle_List[j].position_prediction[0]+50), int(particle_List[j].position_prediction[1]+30))
                cv2.rectangle(img_gray, p1, p2, (255), 2, 1)
                cv2.putText(img_gray,str(particle_List[j].PN) +':' + particle_List[j].type, p1, font, 1, (255), 1, cv2.LINE_AA)
                
            #show summary
        cv2.putText(img_gray, 'algae num:' + str(num_algae) ,(100,950), font, 3, (0), 3, cv2.LINE_AA)
        cv2.putText(img_gray, 'bead num:' + str(num_bead) ,(100,1150), font, 3, (0), 3, cv2.LINE_AA)
    
        cv2.imwrite(f_save,img_gray) 
        continue
    
    
    '''
    detect particles with hog in img
    '''
    dets = detector(img_gray)
    #draw hog detection with black square
    for k, d in enumerate(dets):
        #get bbox from dets
        bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
        #draw_bbox (img_gray, bbox, 0)
    #create a arrary 'hog-new' corresponding to each hog detection
    #0 means this hog detection is new founded
    #1 means this hog match tracking or kalman prediction
    hog_new=np.zeros(len(dets),int)
            
            
    #update tracking and update kalman prediction
    num_particle=len(particle_List)
    for j in range(num_particle):
        '''
        update tracker 
        save position in the class article
        save bbox in the class article
        '''
        particle_List[j].tracker_update(img_gray,frame_num)#update tracker
    # remove the bad tracking.     
    # if movement is more than thorshold, set positon to -100,-100,
    for j in range(num_particle):
        if particle_List[j].bbox[0]>-1: #if tracking j is not removed
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
        
        
<<<<<<< Updated upstream
=======
        #get cost_matrix using prediction, the distance square between prediction and detection
        cost_matrix_predict = np.zeros((len(contours),num_particle))+10000
        for j in range(num_particle):
            if particle_List[j].kalman_ok:
                for i in range(len(contours)):
                    #get position of each contours
                    x_det,y_det= bbox2point(contour_list[i])
                    #get position of each particle
                    x_prediction=particle_List[j].position_prediction[0]
                    y_prediction=particle_List[j].position_prediction[1]
                    #save distance in matrix
                    cost_matrix_predict[i,j]=(x_det-x_prediction)**2+(y_det-y_prediction)**2
                    
                    
    #  get indexs for pair with 1 proity 
        if len(contours)>0 and num_particle>0:
            hog2track=np.zeros(len(contours),int)-1
            hog2predict=np.zeros(len(contours),int)-1
            track2hog=np.zeros(num_particle,int)-1
            predict2hog=np.zeros(num_particle,int)-1
    
            for j in range(num_particle):
                minElement = np.amin(cost_matrix[:,j])
                if minElement < CostOfNonAssignment:
                    track2hog[j] = np.where(cost_matrix[:,j] == minElement)[0][0]
                minElement = np.amin(cost_matrix_predict[:,j])
                if minElement < CostOfNonAssignment:
                    predict2hog[j] = np.where(cost_matrix_predict[:,j] == minElement)[0][0] 
                
            for i in range(len(contours)):
                minElement = np.amin(cost_matrix[i,:])
                if minElement < CostOfNonAssignment:
                    hog2track[i] = np.where(cost_matrix[i,:] == minElement)[0][0] 
                minElement = np.amin(cost_matrix_predict[i,:])
                if minElement < CostOfNonAssignment:
                    hog2predict[i] = np.where(cost_matrix_predict[i,:] == minElement)[0][0] 
              
            #find overlap hog and label fake this hog detection using prediction
            for i in range(len(contours)):
                temp = np.where(predict2hog==i)
                if len(temp[0]) > 1:
                    #release predict pair
                    predict2hog[temp[0]]=-2
                    #report fake hog
                    hog2predict[i]=-2
    
                    hog_fake[i]=1
                    
            #find overlap hog and label fake this hog detection using tracking
            for i in range(len(contours)):
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
            for i in range(len(contours)):
                if hog_fake[i] > 0:
    #                hog_fake[i] = 0
                    cost_matrix=np.delete(cost_matrix, i-i_del, 0)
                    cost_matrix_predict=np.delete(cost_matrix_predict, i-i_del, 0)
                    del contours[i-i_del]
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
    #        for i in range(len(contours)):
    #            for j in range(num_particle):
    #                if cost_matrix_predict[i,j] < CostOfNonAssignment:
    #                    hog_fake[i]=hog_fake[i]+1
    #            if hog_fake[i]>1:
    #                cost_matrix[i,:]=cost_matrix[i,:]*0+10000
    #                cost_matrix_predict[i,:]=cost_matrix_predict[i,:]*0+10000
    #            else:
    #                hog_fake[i]=0
            
    
         
        #fix a bug. case when no detection found. add a fake detection result to let np.amin(cost_matrix[x,x]) run
        if len(contours)==0:
            cost_matrix = np.zeros((1,num_particle))+10000
            cost_matrix_predict = np.zeros((1,num_particle))+10000
        
        #assignDetectionsToTracks(costMatrix,costOfNonAssignment)
        hog_new=np.zeros(len(contours),int)-1
    
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
                        temp=cost_matrix_predict[hog_i,:]###all the cost to one hog detection
                        temp = np.delete(temp, j)
                        minElement_second=5*CostOfNonAssignment#init of second small cost
                        if len(temp)>0:
                            minElement_second = np.amin(temp)#second small cost
                            
                        if minElement < minElement_second and minElement_second > 2*CostOfNonAssignment:
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
                    
         
>>>>>>> Stashed changes
        '''
        Check if results conflicted
        get distance between tracking and hog
        get distance between kalman predict and hog
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

    '''
    Deal with found particles
    '''
    num_particle=len(particle_List)
    for j in range(num_particle):
        if particle_List[j].distance_t2h < D_THOR: 
            if particle_List[j].kalman_ok: #case 1
                particle_List[j].case=1
                #add point
                particle_List[j].report_found()
                #correct kalman filter
                current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
                particle_List[j].kalman.correct(current_position)
                #set flag, hog detection is not new found
                hog_new[particle_List[j].index_t2h]=1
                
                ok = particle_List[j].assign_PN (num_detected)
                if ok:
                    num_detected= num_detected + 1
                
            else: #case 2
                particle_List[j].case=2
                #add point
                particle_List[j].report_found()
                #init kalman filter
                particle_List[j].init_kalman()
                #set flag, hog detection is not new found
                hog_new[particle_List[j].index_t2h]=1

        else:
            if particle_List[j].distance_k2h < D_THOR: #case 3
                particle_List[j].case=3
                #init tracker
                particle_List[j].tracker_create() 
                ok = particle_List[j].tracker.init(img_gray, particle_List[j].hog_bbox)
                particle_List[j].get_save_position(frame_num, particle_List[j].hog_bbox)
                #correct kalman filter
                current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
                particle_List[j].kalman.correct(current_position)
                #set flag, hog detection is not new found
                hog_new[particle_List[j].index_k2h]=1
                
            elif particle_List[j].case==6 and particle_List[j].distance_k2h < 2*D_THOR: #case 3_2
                particle_List[j].case=3
                #init tracker
                particle_List[j].tracker_create() 
                ok = particle_List[j].tracker.init(img_gray, particle_List[j].hog_bbox)
                particle_List[j].get_save_position(frame_num, particle_List[j].hog_bbox)
                #init kalman filter
                particle_List[j].init_kalman()
                #set flag, hog detection is not new found
                hog_new[particle_List[j].index_k2h]=1                
                
                
            elif particle_List[j].distance_k2t < D_THOR: #case 4
                particle_List[j].case=4
                #correct kalman filter
                current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
                particle_List[j].kalman.correct(current_position)
                
            elif not particle_List[j].kalman_ok: #case 5
                particle_List[j].case=5
                particle_List[j].appear = 0

            
            else:#case 6
                particle_List[j].case=6
                particle_List[j].report_missing()
                particle_List[j].save_position(frame_num, particle_List[j].position_prediction)
#                #correct current bbox in the class  with prediction
#                temp_bbox_0= particle_List[j].position_prediction[0]-particle_List[j].bbox[2]/2
#                if temp_bbox_0 < 0:
#                    temp_bbox_0=0
#                
#                temp_bbox_1= particle_List[j].position_prediction[1]-particle_List[j].bbox[3]/2
#                if temp_bbox_1 < 0:
#                    temp_bbox_1=0
#                temp_bbox_2= particle_List[j].bbox[2]
#                temp_bbox_3= particle_List[j].bbox[3]
#                particle_List[j].bbox=(temp_bbox_0,temp_bbox_1,temp_bbox_2,temp_bbox_3)
#                
#                #init tracker
#                particle_List[j].tracker_create() 
#                ok = particle_List[j].tracker.init(img_gray, particle_List[j].bbox)
#                particle_List[j].get_save_position(frame_num, particle_List[j].bbox)

  
    #remove dead object
    point_p=0
    num_particle=len(particle_List)
    for j in range(num_particle):
        if particle_List[point_p].appear == 0:
            del particle_List[point_p]
        else:
            point_p=point_p+1
    
    #case 0
    i_det=0
    for k, d in enumerate(dets):
        if hog_new[i_det]==0:        
            #get bbox from dets        
            bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
            create_new_particle(bbox, img_gray, particle_List, frame_num)
            
        i_det = i_det + 1
    
    # label and summary 
    #print(frame_count)

    num_particle=len(particle_List)
    print('num of tracking object:' + str(num_particle))
    for j in range(num_particle):
        if particle_List[j].appear > 0:
            cv2.circle(img_gray,tuple(particle_List[j].position_prediction), 50, 0, 1)
            #draw_bbox (img_gray, particle_List[j].bbox, 255)
            #cv2.putText(img_gray, str(particle_List[j].case),tuple([particle_List[j].position_x[-1],particle_List[j].position_y[-1]]), font, 1, (255), 1, cv2.LINE_AA)
            cv2.putText(img_gray, str(particle_List[j].PN) + ':' + particle_List[j].type, tuple(particle_List[j].position_prediction), font, 1, (255), 1, cv2.LINE_AA)
            #print(str(j) + '_t2h:' + str(particle_List[j].distance_t2h))
            #print(str(j) + '_k2t:' + str(particle_List[j].distance_k2t))

    #show summary
    cv2.putText(img_gray, 'algae num:' + str(num_algae) ,(100,950), font, 3, (0), 3, cv2.LINE_AA)
    cv2.putText(img_gray, 'bead num:' + str(num_bead) ,(100,1150), font, 3, (0), 3, cv2.LINE_AA)
    
    cv2.imwrite(f_save,img_gray) 
    print('algae num:' + str(num_algae))
    print('bead num:' + str(num_bead))
    print('all num:' + str(num_detected))
    

#print(dir(particle_a))