from collections import deque
import datetime
import sqlite3
import sys
import cv2
import numpy as np
from sort import Sort
import cv2
import time
import os

from threading import Thread

logged_track_ids = set()

class FileFrameFeed(object):
    def __init__(self, source = 0):
        self.capture = cv2.VideoCapture(source)

    def grab_frame(self):
        return self.capture.read()
    
    def is_valid(self):
        return self.capture.isOpened()
    
    def close(self):
        return


class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)
        self.run = True
        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None

    def update(self):
        while self.run:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
        self.capture.release()
        sys.exit()

    def grab_frame(self):
        if self.status:
            return self.status, self.frame
        return None, None
    
    def is_valid(self):
        return self.capture.isOpened()
    
    def close(self):
        self.run = False
        self.thread.join()

def detect_cars(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    boxes_with_confidences = []
    
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
                
                boxes.append([x, y, x + w, y + h])
                boxes_with_confidences.append([x, y, x + w, y + h, confidence])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    cars = []
    for i in range(len(boxes_with_confidences)):
        if i in indexes:
            x, y, w, h, confidence = boxes_with_confidences[i]
            cars.append((x, y, w, h, confidence))
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return cars, frame

def init_db():
    conn = sqlite3.connect("events.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  track_id INTEGER,
                  direction TEXT,
                  filename TEXT)''')
    conn.commit()
    conn.close()

def log_event(track_id, direction, frame_buffer):
    if track_id in logged_track_ids:
        return
    logged_track_ids.add(track_id)
    filename = save_video_clip(list(frame_buffer), track_id)
    conn = sqlite3.connect("events.db")
    c = conn.cursor()
    c.execute('''INSERT INTO events (track_id, direction, filename)
                 VALUES (?, ?, ?)''',
              (track_id, direction, filename))
    conn.commit()
    conn.close()

def save_video_clip(buffer, track_id):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_clip = f"car_{track_id}_{timestamp}.mp4"
    height, width, _ = buffer[0].shape
    out = cv2.VideoWriter(video_clip, fourcc, 20.0, (width, height))

    for frame in buffer:
        out.write(frame)

    out.release()
    return video_clip

print("Initializing database...")
init_db()
print("Initializing video stream...")

endpoint = os.getenv("VIDEO_ENDPOINT")
if endpoint is None:
    print("No endpoint specified, trying the secret file.")
    try:
        with open(".video_endpoint", "r") as f:
            endpoint = f.read().strip()
            print("Found endpoint in secret file!")
    except FileNotFoundError:
        print("No endpoint specified in the secret file either, exiting")
        exit(1)
endpoint = "./samples/sample.mp4"
threaded_camera = ThreadedCamera(endpoint) if endpoint.startswith("rtsp://") else FileFrameFeed(endpoint)

if not threaded_camera.is_valid():
    print("Error: Couldn't open the video stream.")
else:
    print("video stream opened successfully.")

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
frame_buffer = deque(maxlen=100)

i = 0
while True:
    i += 1
    ret, frame = threaded_camera.grab_frame()
    if not ret:
        print("Failed to read frame from video stream.")
        time.sleep(0.1)
        continue

    frame_buffer.append(frame.copy())
    
    cars, frame = detect_cars(frame)
    detections = np.array(cars)
    if detections.size > 0:
        detections = detections.reshape((-1, 5))
        tracks = tracker.update(detections)
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            car_position = car_positions.setdefault(track_id, [])
            
            car_position.append((x1, y1, x2, y2))

            min_threshold = 30
            if len(car_position) > min_threshold:
                x1_old = car_position[-min_threshold][0]
                x1_new = car_position[-1][0]
                delta = x1_new - x1_old
                threshold = 25
                if delta > threshold:
                    direction = "right"
                    log_event(track_id, direction, frame_buffer)
                elif delta < -threshold:
                    direction = "left"
                    log_event(track_id, direction, frame_buffer)
                else:
                    direction = "straight"
                cv2.putText(frame, direction, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, str(track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

threaded_camera.close()
cv2.destroyAllWindows()
