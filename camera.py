import cv2
from threading import Thread


class ThreadedCamera(object):
    def __init__(self, source, consumer):
        self.source = source
        self.capture = cv2.VideoCapture(source)
        self.run = True
        self.consumer = consumer
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def reset(self):
        self.capture = cv2.VideoCapture(self.source)

    def update(self):
        while self.run:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    self.consumer(frame)
                else:
                    self.reset()
            else:
                self.reset()
        self.capture.release()

    def is_valid(self):
        return self.capture.isOpened()

    def close(self):
        self.run = False
        self.thread.join()
