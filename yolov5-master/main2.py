import threading
from multiprocessing import Pool
import time
import cv2
import queue

q = queue.Queue()
cap = cv2.VideoCapture(0)


def process():
    net = cv2.dnn.readNetFromONNX("onnx/yolo5n.onnx")
    img = cap.read()[1]
    img = cv2.resize(img, (320, 320))
    frame = img
    while frame is not None:
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255,
                                     size=(640, 640),
                                     mean=[10, 10, 10],
                                     swapRB=True,
                                     crop=False)
        net.setInput(blob)
        result_of_detection = net.forward()[0]
        q.put(result_of_detection)


def display():
    while True:
        if not q.empty():
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    p1 = threading.Thread(target=process)
    p2 = threading.Thread(target=display)
    p1.start()
    p2.start()
