import cv2
import ball_detector
import time

img = cv2.imread('img.png')

detector = ball_detector.BallDetector()

cap = cv2.VideoCapture()

while True:
    suc, img = cap.read()
    if not suc:
        break
    img = cv2.resize(img, (1280, 720))
    start = time.time()
    rez = detector.detect(img)
    end = time.time()
    dur = end - start
    print(dur, len(rez))

cap.release()