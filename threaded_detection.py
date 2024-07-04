import time
import cv2
import numpy as np
import threading
from queue import Queue

from ultralytics import YOLO
from detection import detect_cars

class ThreadedYOLO:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while True:
            frame = self.frame_queue.get()
            results = self.model(frame, device="mps")
            cars, frame = detect_cars(frame, self.net, self.output_layers, self.classes)
            self.result_queue.put_nowait((cars, frame))

    def queue(self, frame):
        self.frame_queue.put_nowait(frame)

    def get_next(self):
        return self.result_queue.get()
