import cv2 as cv
import numpy as np

img = cv.imread('./cat.jpg')
cv.imshow('cat',img)

# BGR to Gray
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# BGR to HSV
HSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow('HSV', HSV)

# BGR TO L*A*B

lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
cv.imshow('lab',lab)

cv.waitKey(0)