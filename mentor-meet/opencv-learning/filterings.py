import cv2 as cv
import numpy as np

# Gray scale conversion
def gray_scale(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

# Gaussian Blur
def gaussian_blur_image(img, ksize=(7,7)):
    blur = cv.GaussianBlur(img, ksize, sigmaX=5, sigmaY=5, borderType=cv.BORDER_DEFAULT)
    return blur

#Filter 2D
def filter_2D(img,):
    filter_kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    filtered = cv.filter2D(img, kernel=filter_kernel, ddepth=-1)
    return filtered

#Blur
def blur_image(img, ksize=(7,7)):
    blur = cv.blur(img, ksize)
    return blur

#Sharpen
def sharpen_image(img):
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv.filter2D(img, kernel=sharpen_kernel, ddepth=-1)
    return sharpened
