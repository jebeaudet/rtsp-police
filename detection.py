import cv2
import numpy as np

def detect_cars(frame, net, output_layers, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

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

                width = x + w
                height = y + h
                boxes.append([x, y, width, height])
                boxes_with_confidences.append([x, y, width, height, confidence])
                confidences.append(float(confidence))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    cars = []
    for i in range(len(boxes_with_confidences)):
        if i in indexes:
            x, y, w, h, confidence = boxes_with_confidences[i]
            cars.append((x, y, w, h, confidence))

    return cars, frame
