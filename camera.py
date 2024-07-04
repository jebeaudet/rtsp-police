import cv2
from threading import Thread

class ThreadedCamera(object):
    def __init__(self, source, consumer):
        self.capture = cv2.VideoCapture(source)
        self.run = True
        self.consumer = consumer
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

    def update(self):
        while self.run:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    self.consumer(frame)
        self.capture.release()

    def is_valid(self):
        return self.capture.isOpened()

    def close(self):
        self.run = False
        self.thread.join()
