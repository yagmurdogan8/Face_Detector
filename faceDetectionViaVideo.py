# @author YD
# One can detect faces in a live video via using this code
# By enabling the line #11, one can try the code out with already captured videos

import cv2

faceCas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

# cap = cv2.VideoCapture('filename.mp4')

while True:

    _, image = capture.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCas.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('image', image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
