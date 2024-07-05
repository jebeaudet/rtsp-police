import numpy as np
import cv2
from collections import deque
from camera import ThreadedCamera
from database import init_db, log_event
from sort import Sort
from threaded_detection import ThreadedYOLO
from utils import get_video_endpoint

def main():
    init_db()

    yolo = ThreadedYOLO()

    print("Initializing video stream...")
    endpoint = get_video_endpoint()
    endpoint = "./samples/sample-wrong.mp4"
    threaded_camera = ThreadedCamera(endpoint, lambda frame: yolo.queue(frame))

    if not threaded_camera.is_valid():
        print("Error: Couldn't open the video stream.")
        return
    else:
        print("Video stream opened successfully.")

    tracker = Sort()
    car_positions = {}
    frame_buffer = deque(maxlen=100)

    try:
        while True:
            cars, frame = yolo.get_next()
            if cars is None:
                continue
            frame_buffer.append(frame.copy())

            for box, track, track_id in cars:
                x1, y1, x2, y2 = box
                min_threshold = 15
                if len(track) > min_threshold:
                    x1_old = track[-1][0]
                    x1_new = track[0][0]
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

    except KeyboardInterrupt:
        print("Exiting loop!")

    threaded_camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
