import cv2
import numpy as np
from datetime import datetime
import os

# While loop to cature the video of our webcam
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://192.168.1.14:8554/cam") # piCamera
# Dome1: rtsp://admin:kft12345@192.168.1.3:554/stream WP2
# Dome2: rtsp://admin:dome12345@192.168.1.4:554/stream WP1
path = "C:/Users/USER/Desktop/Detections/Sample_Videos/Camera 2 (kft12345).mp4"
cap = cv2.VideoCapture(path)
# Dome3: rtsp://admin:dome54321@192.168.1.2:554/stream WP3
# cap = cv2.VideoCapture("C:/Users/USER/Desktop/Detections/Sample Videos/Free View of Pedestrian Crossing Long - Videvo Free Video Author.mp4") #mp4
# cap = cv2.VideoCapture("rtsp://admin:kft12345@192.168.1.3:554/stream") # WP Stream 2
# cap = cv2.VideoCapture("rtsp://admin:dome12345@192.168.1.4:554/stream") # WP Stream 1
# cap = cv2.VideoCapture("rtsp://admin:dome54321@192.168.1.2:554/stream") # WP Stream 3

#width,height of Target
whT = 320

#setting a confidence Threshold
confThreshold= 0.55

#setting NMS Threshold
nmsThreshold = 0.30


classesFile = 'person.txt'
classNames= []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# modelConfiguration = 'objectDetectionModules\coco_v4\yolov4.cfg' 
# modelWeights = 'objectDetectionModules\coco_v4\yolov4.weights'

"""TINY"""
modelConfiguration = 'E:/person Weights/Version 11/custom-yolov4-tiny-detector.cfg'
modelWeights = 'E:/person Weights/Version 11/custom-yolov4-tiny-detector_final.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



def findObjects(outputs, img):
 
    count = 0
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
#     print(len(bbox))
    # To remove the duplicate boundary boxes we will use NMS
    # Norm Maximum Supression, this elimanates the overlapping boxes
    # it finds the overlapping boxes and based on their confidence values 
    # picks the maximum confidence box and supresses all the non maximum boxes.
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    # the above returns the indices of the bbox to keep
    # So we loop over the indices to define the bbox
#     print(indices)
    for i in indices:
        count = count + 1      
        box = bbox[i] 
        x, y, w, h = box[0], box[1], box[2], box[3]
        # defining the rectangular box
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,255), 2)
        # to display the detected Class ID and confidence
        cv2.putText(img, f'{classNames[classIds[i]].upper()}_{int(confs[i]*100)}',
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255),2)
        cv2.putText(img, f'{"Number of Persons:",len(indices)}%',
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255),2)    
    
   
# Set the initial minute value
previous_minute = datetime.now().minute
while True:
    print("Previoius Minute",previous_minute)
    success, img = cap.read()
    # img = cv2.resize(img, (0, 0), fx=0.5 ,fy=0.5)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    
    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),(0,0,0),
                                 swapRB= True, crop= False)
    net.setInput(blob)
    
    layerNames = net.getLayerNames()    
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    
    outputs = list(net.forward(outputNames))   
    findObjects(outputs, img)
    
    cv2.imshow('KFTImage',img)
#     cv2.waitKey(1)

    dt = datetime.now()   
    values = []
    values.append(f"Person_{dt.strftime('''%d/%m/%y %H:%M''')}")
    directory = 'C:/Users/USER/Desktop/Detections/Sample_Videoss'
    os.makedirs(directory, exist_ok=True)
    
    directory = 'Sample_Videos'
    # filename = 'output_image.jpg'

    # # Concatenate the directory path and filename
    # output_path = directory + filename
    # cv2.imwrite(filename,img)
    # print(values) 
    
     # Get the current minute
    current_minute = datetime.now().minute
    
    print("Current Minutes",current_minute)
    # Check if the minute has changed since the previous save
    if current_minute != previous_minute:
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Specify the filename using the timestamp
        filename = timestamp + '.jpg'

        # Concatenate the directory path and filename
        output_path = directory + filename
        print(output_path)
        # Save the image to the specified directory
        cv2.imwrite(output_path, img)

        # Update the previous minute
        previous_minute = current_minute
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()    