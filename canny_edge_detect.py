from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

max_lowThreshold = 500
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i ][: , 1:]
        XYs = []
        for j in np.unique(npXYs[: , 0]):
            XY = npXYs[npXYs[: , 0] == j ][: , 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)

default_file = 'problems/isolated.csv'
polyline_data = read_csv(default_file)

# Create an empty image
src = np.zeros((250, 250, 3), dtype=np.uint8)

# Draw the polylines onto the image
for path in polyline_data:
    for polyline in path:
        polyline = np.int32(polyline)
        cv.polylines(src, [polyline], isClosed=False, color=(255,255,255))

if src is None:
    print('Could not open or find the file')
    exit(0)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv.waitKey()