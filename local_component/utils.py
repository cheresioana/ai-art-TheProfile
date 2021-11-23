import cv2
import numpy as np


def crop_height(frame, wanted_height=1080):
    height = frame.shape[0]
    start = int((height - wanted_height) / 2)
    end = start + wanted_height
    frame2 = frame[start + 25: end - 100, :]
    return frame2


def showImage(oriimg, w_height=1080, w_width=1920):
    ratio = w_height / oriimg.shape[0]
    w_resize = int(ratio * oriimg.shape[1])
    oriimg = cv2.resize(oriimg, (w_resize, w_height))
    frame = np.zeros((w_height, w_width, 3), dtype="uint8")
    st_x = int((w_width - oriimg.shape[1]) / 2)
    end_x = w_width - st_x
    frame[:, st_x:end_x - 1] = oriimg
    frame = cv2.flip(frame, 1)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("window", frame)


def read_frame_init(cap, w_height=1080):
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = crop_height(frame, wanted_height=w_height)
    return frame

def read_frame(cap, w_height=1080):
    flag, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = crop_height(frame, wanted_height=w_height)
    return frame
