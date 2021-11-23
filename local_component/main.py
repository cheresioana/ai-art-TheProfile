import cv2
import os
import datetime
import time
import numpy as np
import ctypes

from ar.WebcamVideoStream import WebcamVideoStream
from ar.animation import circle1, circle2, circle3
from ar.utils import calculateDistance, calculatePoint
from gan.profile import profile
from remote_communication import set_connection, upload_images, download_latest_weights
from openpose.open_pose_extractorv2 import OpenPoseWrapper
from pact import pact
from person_extractor.person_extractor import person_extraction
from utils import read_frame, showImage, crop_height

w_width = 1920
w_height = 1060

import cv2
import os
import datetime
import time
import numpy as np
import ctypes

from ar.animation import circle1
from ar.utils import calculateDistance, calculatePoint
from openpose.open_pose_extractorv2 import OpenPoseWrapper
from person_extractor.person_extractor import person_extraction
from utils import read_frame, showImage
import imutils
w_width = 1920
w_height = 1060



if __name__ == '__main__':
    ssh_client = set_connection()
    begin_time = datetime.datetime.now()
    start_time = time.time()
    i = 0

    background = None
    vs = WebcamVideoStream(src=0).start()
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    opw = OpenPoseWrapper()
    passed_time = 0
    while (passed_time < 3):
        passed_time = time.time() - start_time
        frame = read_frame(vs)
        showImage(frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        if i == 20:
            cv2.imwrite("doc/background.jpg", frame)
            background = frame
        key = cv2.waitKey(1) & 0xFF
        #time.sleep(0.02)
        i = i + 1
    print(frame.shape)
    #ring
    ring = cv2.imread("ar/ress/ring5.jpg", cv2.IMREAD_UNCHANGED)
    ring = ring[:, :, :3]
    ring = cv2.resize(ring, (frame.shape[1], frame.shape[0]))
    hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([40, 40, 40], np.uint8)
    upper_blue = np.array([70, 255, 255], np.uint8)
    ring_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ring_mask = 255 - ring_mask

    background_ress = cv2.imread("ar/ress/background.jpg")
    background_ress = cv2.resize(background_ress, (frame.shape[1], frame.shape[0]))

    circle1(vs,background, ring, ring_mask, opw, background_ress)
    circle2(vs, background, ring, ring_mask, opw, background_ress)
    circle3(vs, background, ring, ring_mask, opw, background_ress)
    pact(vs, frame, background, opw)
    print("Strated uploading images...")
    upload_images(ssh_client)
    '''print("Timp de uploadare")
    print(datetime.datetime.now() - begin_time)
    circle2(vs, frame, background, ring, opw)
    circle3(vs, frame, background, ring, opw)
    pact(vs, frame, background, ring, opw)
    print("Started downloading weights...")'''
    #download_latest_weights(ssh_client)
    print("Timp de download weight-uri")
    '''print(datetime.datetime.now() - begin_time)
    profile(vs, frame, background,  opw)



    cv2.destroyAllWindows()
    vs.stop()
    print(i)'''

