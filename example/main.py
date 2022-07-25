import cv2
import ball_detector

img = cv2.imread('img.png')

balls = ball_detector.detect(img)

for ball in balls:
    cv2.circle(img, (ball[0] + ball[2]//2, ball[1] + ball[3]//2), ball[2]//2, (0, 0, 255), 5)

cv2.imshow('ball', img)
cv2.waitKey(0)