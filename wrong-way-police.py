import cv2
import numpy as np
from sort import Sort
import cv2
import time
import os

def detect_cars(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    cars = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cars.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return cars, frame

print("Initializing RTSP stream...")

endpoint = os.getenv("RTSP_ENDPOINT")
if endpoint is None:
	print("No endpoint specified, exiting!")
	exit(1)
cap = cv2.VideoCapture(endpoint)

if not cap.isOpened():
    print("Error: Couldn't open the RTSP stream.")
else:
    print("RTSP stream opened successfully.")

# Load YOLO model
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
# Fix to get output layer names
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:  # Handle the scalar issue
    output_layers = [layer_names[net.getUnconnectedOutLayers() - 1]]

with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

tracker = Sort()

car_positions = {}

while True:
    ret, frame = cap.read()
    print("processing frame")
    if not ret:
        print("Failed to read frame from RTSP stream.")
        break
    
    cars, frame = detect_cars(frame)
    detections = np.array(cars)
    tracks = tracker.update(detections)
    
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        if track_id not in car_positions:
            car_positions[track_id] = []
        
        car_positions[track_id].append((x1, y1, x2, y2))
        
        if len(car_positions[track_id]) > 1:
            x1_old, y1_old, x2_old, y2_old = car_positions[track_id][-2]
            x1_new, y1_new, x2_new, y2_new = car_positions[track_id][-1]
            direction = "right" if x1_new > x1_old else "left"
            cv2.putText(frame, direction, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, str(track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    #cv2.imshow('Frame', frame)
    
   # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()

