import cv2 as cv
import numpy as np
import rescale 
import filterings
import edge_detectors as ed
import thresholding as th
import morpological as mp

#Converting image to gray scale

img = cv.imread("./Photos/shepd.jpg")
cv.imshow("Original Image", img)

# gray_image = filterings.gray_scale(img)
# cv.imshow("Gray Image", gray_image)

# # Blur the image Gsussian Blur

# blurred_image = filterings.gaussian_blur_image(img,(9,9))
# cv.imshow("Blurred Image", blurred_image)

# Apply the custom kernel to the image - filter2D
# filtered_image = filterings.filter_2D(img)
# cv.imshow("Filtered Image", filtered_image)

# # Blur the image using the blur function
# blurred_img = filterings.blur_image(img, (10,10))
# cv.imshow("Blurred Image", blurred_img)

# # Sharpen the image
# sharpened_image = filterings.sharpen_image(img)
# cv.imshow("Sharpened Image", sharpened_image)

 ###  Edge Detection ###

 #sobel edge edetection
# ed.sobel_edge_detectors(img)

# # laplacian edge detection
# laplacian_image = ed.laplacian_edge_detectors(img)
# cv.imshow("Laplacian Image", laplacian_image)

# # canny edge detection
# canny_image = ed.canny_edge_detectors(img, 100, 200)
# cv.imshow("Canny Image", canny_image)
# cannay2 = ed.canny_edge_detectors(img, 20, 60)
# cv.imshow("Canny Image 2", cannay2)

# canny edge detection for video
ed.canny_edge_detectors_video('./Videos/dog-cat.mp4', 100, 200)

###  Thresholding : 

# #Global Thresholding

# global_thresh_image = th.global_thresholding(img)
# cv.imshow("Global Thresholding", global_thresh_image)

# Otsu Thresholding

# otsu_thresh_image = th.otsu_thresholding(img)
# cv.imshow("Otsu Thresholding", otsu_thresh_image)

# # Adaptive Thresholding
# adaptive_thresh_image = th.adaptive_thresholding(img)
# cv.imshow("Adaptive Thresholding", adaptive_thresh_image)

# # ## Erosion 
# eroded_image = mp.erosion(img)
# cv.imshow("Eroded Image", eroded_image)

# ## Dilation
# dilated_image = mp.dilation(img)
# cv.imshow("Dilated Image", dilated_image)

# ## Opening
# opening_image = mp.opening(img)
# cv.imshow("Opening Image", opening_image)

# ## Closing
# closing_image = mp.closing(img)
# cv.imshow("Closing Image", closing_image)

cv.waitKey(0)