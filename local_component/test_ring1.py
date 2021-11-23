import tensorflow as tf
import cv2
import os

from ar.WebcamVideoStream import WebcamVideoStream
from ar.animation import add_layer
from gan.HandGan import resize, normalize
from gan.Loader import Loader
from person_extractor.person_extractor import person_extraction2
from openpose.open_pose_extractorv2 import OpenPoseWrapper
import math
from part_extractor.part_extractor import get_hand
import numpy as np
import datetime
import paramiko
import gc
from utils import showImage, read_frame

background_file = 'doc/background.jpg'
dataset_folder = './doc/mini_dataset8'


if __name__ == '__main__':
    begin_time = datetime.datetime.now()


    i = 0
    opw = OpenPoseWrapper()

    vs = cv2.VideoCapture("doc/video2.mp4")
    background_ress = cv2.imread("ar/ress/background.jpg")
    isis_res = cv2.imread("ar/ress/isis-2.png")
    isis_res = cv2.flip(isis_res, 1)
    ring = cv2.imread("ar/ress/ring5.jpg", cv2.IMREAD_UNCHANGED)
    ring = ring[:, :, :3]
    lower_blue = np.array([40, 40, 40], np.uint8)
    upper_blue = np.array([70, 255, 255], np.uint8)


    while True:

        frame = read_frame(vs)

        key, skeleton = opw.tag_person(frame)
        if i == 0:
            background_ress = frame
            black = np.zeros_like(background_ress)

            ring = cv2.resize(ring, (frame.shape[1], frame.shape[0]))
            hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
            ring_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            ring_mask = 255 - ring_mask

            black[:, :] = (10, 0, 0)
            dark_ring = cv2.addWeighted(ring, 0.6, black, 0.4, 0, ring_mask)
            dark_background = cv2.addWeighted(background_ress, 0.05, black, 0.95, 0)

        elif i > 80:

            person2 = person_extraction2(frame, background_ress)
            gray_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
            gray_person2 = cv2.cvtColor(person2, cv2.COLOR_BGR2GRAY)
            mask = ring_mask.copy()
            mask[np.where(gray_skeleton > 10)] = 0

            skeleton[np.where(mask > 0)] = dark_ring[np.where(mask > 0)]

            mask[np.where(gray_skeleton > 10)] = 255
            mask = 255 - mask

            skeleton[np.where(mask > 0)] = dark_background[np.where(mask > 0)]

            cv2.imshow('person_before_layer', skeleton)
            person = add_layer(isis_res, skeleton, frame, gray_person2, 1)
            cv2.imshow('person2', person)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        i = i + 1

