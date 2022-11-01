import cv2
import ball_detector

img = cv2.imread('img.png')

detector = ball_detector.BallDetector()

cap = cv2.VideoCapture('C:/Users/User/Downloads/DSC_8918.MP4')

while True:
    suc, img = cap.read()
    img = cv2.resize(img, (640, 360))
    if not suc:
        break
    rez = detector.detect(img)
    print(rez)