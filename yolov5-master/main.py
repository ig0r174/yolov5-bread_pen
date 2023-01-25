import time

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromONNX("onnx/yolo5n.onnx")
# net = cv2.dnn.readNetFromONNX("yolov5-master/onnx/yolo5s.onnx")
file = open("onnx/pen_bread.txt", "r")
classes = file.read().split('\n')
print(classes)

prev_frame_time = 0
new_frame_time = 0
increment = 0
cachedDetections = None

while True:
    img = cap.read()[1]
    frame = img
    if img is None:
        break

    img = cv2.resize(img, (640, 640))
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(640, 640), mean=[10, 10, 10], swapRB=True, crop=False)
    net.setInput(blob)
    detections = cachedDetections
    if increment % 3 == 1:
        detections = net.forward()[0]
        cachedDetections = detections.copy()

    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    if detections is not None:
        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]

        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width / 640
        y_scale = img_height / 640

        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.5:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.5:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx - w / 2) * x_scale)
                    y1 = int((cy - h / 2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1, y1, width, height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        for i in indices:
            x1, y1, w, h = boxes[i]
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = label + "{:.2f}".format(conf)
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)


    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("FRAME", img)
    increment += 1
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()