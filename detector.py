# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:04:41 2021

@author: 82102
"""

import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
from sklearn import metrics
import matplotlib.pyplot as plt
import pygame
import time

knn = cv2.ml.KNearest_create()

def start(sample_size=25) :
    train_data = generate_data(sample_size)
    #print("train_data :",train_data)
    labels = classify_label(train_data)
    power, nomal, short = binding_label(train_data, labels)
    print("Return true if training is successful :", knn.train(train_data, cv2.ml.ROW_SAMPLE, labels))
    return power, nomal, short

def run(new_data, power, nomal, short):
    a = np.array([new_data])
    b = a.astype(np.float32)
    #plot_data(power, nomal, short)    
    ret, results, neighbor, dist = knn.findNearest(b, 5) # Second parameter means 'k'
    #print("Neighbor's label : ", neighbor)
    print("predicted label : ", results)
    #print("distance to neighbor : ", dist)
    #print("what is this : ", ret)
    #plt.plot(b[0,0], b[0,1], 'm*', markersize=14);
    return int(results[0][0])
    
#'num_samples' is number of data points to create
#'num_features' means the number of features at each data point (in this case, x-y conrdination values)
def generate_data(num_samples, num_features = 2) :
    """randomly generates a number of data points"""    
    data_size = (num_samples, num_features)
    data = np.random.randint(0,40, size = data_size)
    return data.astype(np.float32)

#I determined the drowsiness-driving-risk-level based on the time which can prevent driving accident.
def classify_label(train_data):
    labels = []
    for data in train_data :
        if data[1] < data[0]-15 :
            labels.append(2)
        elif data[1] >= (data[0]/2 + 15) :
            labels.append(0)
        else :
            labels.append(1)
    return np.array(labels)

def binding_label(train_data, labels) :
    power = train_data[labels==0]
    nomal = train_data[labels==1]
    short = train_data[labels==2]
    return power, nomal, short

def plot_data(po, no, sh) :
    plt.figure(figsize = (10,6))
    plt.scatter(po[:,0], po[:,1], c = 'r', marker = 's', s = 50)
    plt.scatter(no[:,0], no[:,1], c = 'g', marker = '^', s = 50)
    plt.scatter(sh[:,0], sh[:,1], c = 'b', marker = 'o', s = 50)
    plt.xlabel('x is second for alarm term')
    plt.ylabel('y is 10s for time to close eyes')

#We don't use below two functions. 
def accuracy_score(acc_score, test_score) :
    """Function for Accuracy Calculation"""
    print("KNN Accuracy :",np.sum(acc_score == test_score) / len(acc_score))
    #A line below this comment is exactly same with above one.
    #print(metrics.accuracy_score(acc_score, test_score))
    
def precision_score(acc_score, test_score) :
    """Function for Precision Calculation"""
    true_two = np.sum((acc_score == 2) * (test_score == 2))
    false_two = np.sum((acc_score != 2) * (test_score == 2))
    precision_two = true_two / (true_two + false_two)
    print("Precision for the label '2' :", precision_two)
    
    true_one = np.sum((acc_score == 1) * (test_score == 1))
    false_one = np.sum((acc_score != 1) * (test_score == 1))
    precision_one = true_one / (true_one + false_one)
    print("Precision for the label '1' :", precision_one)
    
    true_zero = np.sum((acc_score == 0) * (test_score == 0))
    false_zero = np.sum((acc_score != 0) * (test_score == 0))
    precision_zero = true_zero / (true_zero + false_zero)
    print("Precision for the label '0' :", precision_zero)
    
def light_removing(frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]
    med_L = cv2.medianBlur(L,99) #median filter
    invert_L = cv2.bitwise_not(med_L) #invert lightness
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
    return L, composed

def select_alarm(result) :
    if result == 0:
        sound_alarm("power_alarm.wav")
    elif result == 1 :
        sound_alarm("nomal_alarm.wav")
    else :
        sound_alarm("short_alarm.mp3")

def sound_alarm(path) :
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
def init_open_ear() :
    time.sleep(5)
    print("open init time sleep")
    ear_list = []
    th_message1 = Thread(target = init_message)
    th_message1.deamon = True
    th_message1.start()
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)
    print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")

def init_close_ear() : 
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    ear_list = []
    th_message2 = Thread(target = init_message)
    th_message2.deamon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR) #EAR_THRESH means 50% of the being opened eyes state
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :",EAR_THRESH, "\n")

def init_message() :
    print("init_message")
    sound_alarm("init_sound.mp3")

def check_fps(prev_time) :
    cur_time = time.time() #Import the current time in seconds
    one_loop_time = cur_time - prev_time
    prev_time = cur_time
    fps = 1/one_loop_time
    return prev_time, fps

#####################################################################################################################
#1. Variables for checking EAR.
#2. Variables for detecting if user is asleep.
#3. When the alarm rings, measure the time eyes are being closed.
#4. When the alarm is rang, count the number of times it is rang, and prevent the alarm from ringing continuously.
#5. We should count the time eyes are being opened for data labeling.
#6. Variables for trained data generation and calculation fps.
#7. Detect face & eyes.
#8. Run the cam.
#9. Threads to run the functions in which determine the EAR_THRESH. 

#1.
OPEN_EAR = 0 #For init_open_ear()
EAR_THRESH = 0 #Threashold value

#2.
#It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
EAR_CONSEC_FRAMES = 22
COUNTER = 0 #Frames counter.

#3.
closed_eyes_time = [] #The time eyes were being offed.
TIMER_FLAG = False #Flag to activate 'start_closing' variable, which measures the eyes closing time.
ALARM_FLAG = False #Flag to check if alarm has ever been triggered.

#4. 
ALARM_COUNT = 0 #Number of times the total alarm rang.
RUNNING_TIME = 0 #Variable to prevent alarm going off continuously.

#5.    
PREV_TERM = 0 #Variable to measure the time eyes were being opened until the alarm rang.

#6. make trained data 
np.random.seed(9)
power, nomal, short = start(25) #actually this three values aren't used now. (if you use this, you can do the plotting)
#The array the actual test data is placed.
test_data = []
#The array the actual labeld data of test data is placed.
result_data = []
#For calculate fps
prev_time = 0

#7. 
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#8.
print("starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

#9.
th_open = Thread(target = init_open_ear)
th_open.deamon = True
th_open.start()
th_close = Thread(target = init_close_ear)
th_close.deamon = True
th_close.start()

#####################################################################################################################

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 600)
    
    L, gray = light_removing(frame)
    
    rects = detector(gray,0)
    
    #checking fps. If you want to check fps, just uncomment below two lines.
    #prev_time, fps = check_fps(prev_time)
    #cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #(leftEAR + rightEAR) / 2 => both_ear. 
        both_ear = (leftEAR + rightEAR) * 500  #I multiplied by 1000 to enlarge the scope.

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        

        if both_ear < EAR_THRESH :
            if not TIMER_FLAG:
                start_closing = timeit.default_timer()
                TIMER_FLAG = True
            COUNTER += 1

            if COUNTER >= EAR_CONSEC_FRAMES:

                mid_closing = timeit.default_timer()
                closing_time = round((mid_closing-start_closing),3)

                if closing_time >= RUNNING_TIME:
                    if RUNNING_TIME == 0 :
                        CUR_TERM = timeit.default_timer()
                        OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM),3)
                        PREV_TERM = CUR_TERM
                        RUNNING_TIME = 1.75
                      

                    RUNNING_TIME += 2
                    ALARM_FLAG = True
                    ALARM_COUNT += 1

                    print("{0}st ALARM".format(ALARM_COUNT))
                    print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME)
                    print("closing time :", closing_time)
                    test_data.append([OPENED_EYES_TIME, round(closing_time*10,3)])
                    result = run([OPENED_EYES_TIME, closing_time*10], power, nomal, short)
                    result_data.append(result)
                    t = Thread(target = select_alarm, args = (result, ))
                    t.deamon = True
                    t.start()

        else :
            COUNTER = 0
            TIMER_FLAG = False
            RUNNING_TIME = 0

            if ALARM_FLAG :
                end_closing = timeit.default_timer()
                closed_eyes_time.append(round((end_closing-start_closing),3))
                print("The time eyes were being offed :", closed_eyes_time)

            ALARM_FLAG = False

        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    cv2.imshow("detector",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()