import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import count
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


def real_time_detection(model, classLabels):
    cap = cv2.VideoCapture('videos/videoplayback.mp4') # type 0 for live webcam feed detection
    spatial_info = pd.DataFrame({'Person Count':[0],
                                  'Activity Indicator':[False]})
    spatial_info.to_csv('Data Files/spatial.csv',sep=',',index=True, index_label='Time') # Begin with empty csv file
    while True:
        ret, frame = cap.read()
        if ret == False: # Video ended
            break

        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)
        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN
        people_in_frame = Person_count(ClassesPresent=ClassIndex, ClassLabels=classLabels)
        spatial_info.loc[len(spatial_info.index)] = [people_in_frame, False] # update the dataframe
        spatial_info.to_csv(path_or_buf='Data Files/spatial.csv',sep=',',index=True, index_label='Time') # update the csv
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
    real_time_detection(model=model, classLabels=classLabels)
    