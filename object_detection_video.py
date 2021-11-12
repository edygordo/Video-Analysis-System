import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpu.string import str2bool
from multiprocessing import Process, Value, Array, Lock
import sys
from matplotlib.animation import FuncAnimation

def load_pretrained_model():
    config_file = 'Object Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # configuration file
    frozen_model = 'Object Detection/frozen_inference_graph.pb' # model
    model = cv2.dnn_DetectionModel(frozen_model, config_file) # loaded model in memory
    classLabels = [] # List of class Labels
    file_name = 'Object Detection/Labels.txt'
    with open(file_name,'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')
    return model, classLabels

def print_labels(classLabels,verbose=0):
    print(f'This Model can detect a total of :- {len(classLabels)} Labels.')
    if verbose == 1:
        for idx,labels in enumerate(classLabels):
            print(f'Label {idx+1}. {classLabels[idx]}')

def run_animation(model, classLabels):
    ani = FuncAnimation(plt.gcf(), live_person_count, interval=1000)
    plt.show()

def live_person_count(frame):
    data_file = pd.read_csv('Data Files/spatial.csv',sep=',',index_col=0) # read the csv
    x_vals = data_file.index.values # get x values
    y_vals_people = data_file['Person Count'].values # get person count values
    y_vals_motion = data_file['Activity Indicator']
    plt.cla()
    plt.plot(x_vals, y_vals_people, label='Live Person Count')
    plt.fill_between(x_vals, y_vals_motion, alpha=0.5, step='pre', color='r')
    plt.step(x_vals, y_vals_motion, label='Motion Detection', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()

def setInputParams(model, width=320, height=320):
    model.setInputSize(width,height) # input configuration file defined this as the input size
    model.setInputScale(1.0/127.5) # 255(all gray levels)/2
    model.setInputMean((127.5,127.5,127.5)) # mobilenet=>[-1, 1]
    model.setInputSwapRB(True) # input in RGB format to model

def Person_count(ClassesPresent, ClassLabels): # Returns person present given ClassIndex present and all class labels
    Person_present = 0
    if len(ClassesPresent) == 0:
        return Person_present
    else: # Some classes are present
        for idx in ClassesPresent:
            if ClassLabels[idx-1] == 'person':
                Person_present = Person_present + 1
            else:
                pass
    return Person_present

def motion_present(frame1, frame2, motion_thresh=900):
    diff = cv2.absdiff(frame1, frame2) # Find difference between successive frames
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # color the diff image Gray
    blur = cv2.GaussianBlur(gray, (5,5), 0) # apply a gaussian blur to extract contour
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # For Contour extraction
    dilated = cv2.dilate(thresh, None, iterations=3) # Image Processing technique for contour extraction
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find all the contours present in frame 1

    for contour in contours: # iterate over all the contours in a frame1
        if cv2.contourArea(contour) < motion_thresh: # apply area threshold to detect People among all contours
            continue
        else:
            return True # some motion is present
    return False # no motion present in frame

def real_time_detection(model, classLabels, video_src='videos/street_video_1.mp4'): # this file generates a csv datafile in REAL TIME
    cap = cv2.VideoCapture(video_src) # type 0 for live webcam feed detection
    spatial_info = pd.DataFrame({'Person Count':[0],
                                  'Activity Indicator':[False]})
    spatial_info.to_csv('Data Files/spatial.csv',sep=',',index=True, index_label='Frame Number') # Begin with empty csv file
    while True:
        ret, frame = cap.read()
        cap_v = cap
        _, frame2 = cap_v.read()
        if ret == False: # Video ended
            break

        
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)
        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN
        people_in_frame = Person_count(ClassesPresent=ClassIndex, ClassLabels=classLabels)
        Motion = motion_present(frame1=frame, frame2=frame2)
        spatial_info.loc[len(spatial_info.index)] = [people_in_frame, Motion] # update the dataframe
        spatial_info.to_csv(path_or_buf='Data Files/spatial.csv',sep=',',index=True, index_label='Time') # update the csv

        # data_file = pd.read_csv('Data Files/spatial.csv',sep=',',index_col=0) # read the csv
        # x_vals = data_file.index.values # get x values
        # y_vals_people = data_file['Person Count'].values # get person count values
        # plt.cla()
        # plt.plot(x_vals, y_vals_people, label='Live Person Count')
        # plt.legend(loc='upper left')
        # plt.tight_layout()


        if(len(ClassIndex) != 0):
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                cv2.rectangle(frame, boxes, color=(255,0,0),thickness=2 )
                cv2.putText(frame, classLabels[ClassInd-1],(boxes[0]+10, boxes[1]+40), font, fontScale=font_scale,color=(0,255,0))
        else:
            pass
        cv2.imshow('Real Time object detection using MobileNet SSD', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    model, classLabels = load_pretrained_model() # Load a pre-trained model
    setInputParams(model=model) # Set input parameters to the model
    #ani = FuncAnimation(plt.gcf(), live_person_count, interval=1000)
    
    #ani = FuncAnimation(plt.gcf(), live_person_count,interval=1000)
    # p1 = Process(target=real_time_detection(model,classLabels))
    # p1.start()
    # p2 = Process(target=run_animation(model, classLabels))
    # p2.start()
    #real_time_detection(model=model, classLabels=classLabels) # start generating a live csv file
    #ani = FuncAnimation(plt.gcf(), live_person_count, interval=1000, frames=1000)
    # real_time detection and run_animation must run parallely
    p1 = Process(target=real_time_detection(model, classLabels))
    p2 = Process(target=run_animation(model, classLabels))

    p2.start()
    p1.start()

    p1.join()
    p2.join()

    plt.show()