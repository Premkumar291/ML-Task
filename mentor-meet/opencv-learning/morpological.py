import cv2 as cv
import numpy as np
import thresholding as th
import filterings

## Erosion
def erosion(img):
    kernel = np.ones((5,5),np.uint8)

    binary_img = th.global_thresholding(img)
    eroded_img = cv.erode(binary_img,kernel)
    return eroded_img

## Dilation
def dilation(img):
    kernel = np.ones((5,5),np.uint8)

    binary_img = th.global_thresholding(img)
    dilated_img = cv.dilate(binary_img,kernel)
    return dilated_img

## opening
def opening(img):
    kernel = np.ones((5,5),np.uint8)

    binary_img = th.global_thresholding(img)
    opening_img = cv.morphologyEx(binary_img,cv.MORPH_OPEN,kernel)
    return opening_img

## closing
def closing(img):
    kernel = np.ones((5,5),np.uint8)

    binary_img = th.global_thresholding(img)
    closing_img = cv.morphologyEx(binary_img,cv.MORPH_CLOSE,kernel)
    return closing_img