# @author YD
# One can detect faces in a photo file via using this code
# By changing the name of the image file (line #10), one can try the code out with different images

import numpy
import cv2

faceCas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('groupOfPeople.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCas.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('image', image)
cv2.waitKey()
