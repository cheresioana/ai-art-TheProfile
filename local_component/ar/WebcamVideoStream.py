from threading import Thread
import cv2

from ar.utils import calculateDistance
from part_extractor.part_extractor import get_body
import numpy as np

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self.index = 0

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
    def save_train(self, pose, image, keypoints):
        self.index = self.index + 1
        Thread(target=self.save_image, args=(pose, image, keypoints, self.index)).start()

        #self.save_image(pose, image, keypoints, self.index)
    def save_image(self, pose, image, keypoints, index):
        image = cv2.resize(image, (256, 256))
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.dilate(image, kernel, iterations=1)
        pose = cv2.resize(pose, (256, 256))
        self.index = self.index + 1
        cv2.imwrite("dataset/base_image" + str(self.index) + ".jpg", image)
        final_image = cv2.hconcat([pose, image])
        #cv2.imshow('final_image', final_image)
        cv2.imwrite("dataset/train/" + str(index) + ".jpg", final_image)
        #head, body = get_body(keypoints, image)
        #frame = image
        '''if not keypoints is None:
            for i in keypoints:
                p2_x, p3_x, p2_y, p3_y = (0, 0, 0, 0)
                if (i[16][2] > 0.3):
                    p2_x = i[16][0]
                    p2_y = i[16][1]
                    p3_x = i[17][0]
                    p3_y = i[17][1]
                    p0_x = i[0][0]
                    p0_y = i[0][1]
                    p1_x = i[1][0]
                    p1_y = i[1][1]
                    ux = i[2][0]
                    uy = i[2][1]
                    height = frame.shape[0]
                    width = frame.shape[1]
                    mask = np.zeros((height, width), dtype=np.uint8)
                    center_x = int((p2_x + p3_x) / 2)
                    center_y = int((p2_y + p3_y) / 2)
                    horizontal = calculateDistance(p2_x, p2_y, p3_x, p3_y)
                    vertical = calculateDistance(p0_x, p0_y, p1_x, p1_y)
                    axesLength = (int(horizontal), int(2 * vertical))
                    mask = cv2.ellipse(mask, (center_x, center_y), axesLength, 0, 0, 360, (255, 255, 255), -1,
                                       cv2.LINE_AA)
                    head = cv2.bitwise_and(frame, frame, mask=mask)
                    inv_mask = 255 - mask
                    inv_mask[int(uy) - 10:, int(ux) - 10:] = 255
                    body = cv2.bitwise_and(frame, frame, mask=inv_mask)
            cv2.imshow('headq', head)
            cv2.imshow('body', body)'''



