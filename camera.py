import cv2
import sys
from threading import Thread

class FileFrameFeed(object):
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)

    def grab_frame(self):
        return self.capture.read()

    def is_valid(self):
        return self.capture.isOpened()

    def close(self):
        self.capture.release()


class ThreadedCamera(object):
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.run = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

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
