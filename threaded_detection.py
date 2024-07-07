import time
from collections import defaultdict
from typing import List

import cv2
import numpy as np
import threading
from queue import Queue

from ultralytics import YOLO
from ultralytics.engine.results import Results

from bounded_dict import BoundedDict


class ThreadedYOLO:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.track_history = BoundedDict(200, lambda: [])
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while True:
            frame = self.frame_queue.get()
            results = self.model.track(frame, persist=True, device="mps", conf=0.5, classes=[2, 3, 5, 7], verbose=False)
            cars = self.process_results(results)
            if not cars:
                continue
            self.result_queue.put_nowait((cars, frame))

    def queue(self, frame):
        self.frame_queue.put_nowait(frame)

    def get_next(self):
        return self.result_queue.get()

    def process_results(self, results: List[Results]):
        car_detections = []
        raw_boxes = results[0].boxes
        if len(raw_boxes) == 0:
            return None
        boxes = raw_boxes.xyxy.cpu()
        track_ids = raw_boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            track = self.track_history[track_id]
            track.append((float(x1), float(y1)))
            car_detections.append(([x1, y1, x2, y2], track, track_id))

        return car_detections
