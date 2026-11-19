import cv2 as cv
import numpy as np
import filterings

#sobel Edge detection
def sobel_edge_detectors(img):
    gray_img = filterings.gray_scale(img)
    sobel_x = cv.Sobel(gray_img,cv.CV_64F,1,0,ksize=3)
    sobel_y = cv.Sobel(gray_img,cv.CV_64F,0,1,ksize=5)

    abs_sobel_x = cv.convertScaleAbs(sobel_x)
    abs_sobel_y = cv.convertScaleAbs(sobel_y)

    combined_addWeight = cv.addWeighted(abs_sobel_x,0.5,abs_sobel_y,0.5,0)
    coimbined_magitude = cv.magnitude(sobel_x,sobel_y)
    abs_combined_magnitude = cv.convertScaleAbs(coimbined_magitude)

    # cv.imshow("Sobel X",sobel_x)
    # cv.imshow("Sobel Y",sobel_y)
    # cv.imshow("Combined",combined_addWeight)
    cv.imshow("Magnitude",abs_combined_magnitude)

#laplacian edge detection
def laplacian_edge_detectors(img):
    gray_img = filterings.gray_scale(img)
    laplacian = cv.Laplacian(gray_img,cv.CV_64F)
    return laplacian

#canny edge detectors
def canny_edge_detectors(img,threshold1,threshold2):
    gray_img = filterings.gray_scale(img)
    canny = cv.Canny(gray_img,threshold1,threshold2)
    return canny

#canny based video detection
def canny_edge_detectors_video(video_path,threshold1,threshold2):
    capture = cv.VideoCapture(0)
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        gray_frame = filterings.gray_scale(frame)
        canny_frame = cv.Canny(gray_frame,threshold1,threshold2)
        cv.imshow("Canny Video",canny_frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break