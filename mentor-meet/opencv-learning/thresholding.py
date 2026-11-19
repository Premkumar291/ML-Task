import cv2 as cv
import numpy as np
import filterings

###  Thresholding : 

#Global Thresholding
def global_thresholding(img):
    gray_img = filterings.gray_scale(img)
    _, thresh = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
    return thresh

#Otsu Thresholding
def otsu_thresholding(img):
    gray_img = filterings.gray_scale(img)
    _, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

#Adaptive Thresholding
def adaptive_thresholding(img):
    gray_img = filterings.gray_scale(img)
    thresh = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    return thresh