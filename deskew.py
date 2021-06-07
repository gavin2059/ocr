"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
import os

def hough(image):
    dst = cv.Canny(image, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    # lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    #     cv.imwrite('cdst' + default_file, cdst)
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 75, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            cv.imwrite('cdstP.jpg', cdstP)
    print(linesP)
    return linesP

def findAngle(lines):
    angles = {}
    for i in range(0, len(lines)):
        line = lines[i][0]
        slope = (line[3] - line[1]) / (line[2] - line[0])
        angle = math.atan(slope) * 180 / math.pi
        angle = angle - (angle % 5)
        if angle in angles:
            angles[angle] += 1
        else:
            angles[angle] = 1
    print(angles)
    return max(angles, key = angles.get)

def rotate(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv.warpAffine(newImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return newImage

def correct(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # _, bw = cv.threshold(gray, 75, 255, cv.THRESH_BINARY)
    # cv.imwrite('bw.jpg', bw)
    lines = hough(gray)
    angle = findAngle(lines)
    print(angle)
    return rotate(image, -1 * angle)

if __name__ == "__main__":
    default_file = 'e1.jpg'
    filename = sys.argv[0] if len(sys.argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
    else:
        lines = hough(src)
        print(findAngle(lines))