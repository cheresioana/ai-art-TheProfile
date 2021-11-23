from __future__ import print_function

import cv2 as cv
import numpy as np

def person_extraction2(frame, background):
    mask1 = cv.absdiff(frame, background)
    mask = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY)
    th = 30
    imask = mask > th
    canvas2 = np.zeros_like(frame, np.uint8)
    canvas2[imask] = 255
    canvas = np.zeros_like(frame, np.uint8)
    canvas[imask] = frame[imask]

    return canvas

