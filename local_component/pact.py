import cv2
import os
import datetime
import time
import numpy as np
import ctypes

from ar.utils import calculateDistance, calculatePoint

from utils import showImage, read_frame


w_width = 1916
w_height = 1080

x_coords = 100
y_corrds = 600
alpha = 0.4
first_time = 1
p3x, p3y, p4x, p4y, p6x, p6y, p7x, p7y = np.zeros(8)
circle_color = (255, 255, 255)
res_shape0, res_shape1 = (0, 0)
glob_stage  = 0

def morph_trans(mask):
    dilatation_size = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    mask = cv2.erode(mask, element)
    mask = cv2.dilate(mask, element)
    dilatation_size = 10
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    mask = cv2.dilate(mask, element)
    return mask

def alphaBlend(img1, img2, mask):
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

def add_layer(res, person, frame, gray_person, stage):
    global res_shape0, res_shape1
    if stage == 0:
        res_shape0 = res_shape0 + 30
        res_shape1 = res_shape1 + 30
        if res_shape1 > res.shape[1]:
            res_shape1 = res.shape[1]
        if res_shape0 > res.shape[0]:
            res_shape0 = res.shape[0]
    elif stage == 1:
        res_shape0 = res.shape[0]
        res_shape1 = res.shape[1]
    elif stage == 2:
        res_shape0 = res_shape0 - 30
        res_shape1 = res_shape1 - 30
        if res_shape0 < 0:
            res_shape0 = 0
        if res_shape1 < 0:
            res_shape1 = 0

    mask = np.zeros_like(frame)
    mask1 = np.zeros_like(frame)
    layer2 = np.zeros_like(frame)

    blend_mask = np.zeros_like(frame, dtype=np.uint8)

    layer2[x_coords: x_coords + res_shape0, y_corrds: y_corrds + res_shape1] = res[:res_shape0, :res_shape1]
    mask1[x_coords: x_coords + res_shape0, y_corrds: y_corrds + res_shape1] = 255

    cv2.circle(blend_mask, (  int((2 * y_corrds + res_shape1) / 2), int((2 * x_coords + res_shape0) / 2)),
               int(max(res_shape0, res_shape1) / 2)
               , (255, 255, 255), -1, cv2.LINE_AA)
    blend_mask = cv2.GaussianBlur(blend_mask, (121, 121), 11)
    blured = cv2.GaussianBlur(layer2, (21, 21), 11)
    blended2 = alphaBlend(layer2, blured, 255 - blend_mask)
    blended2 = alphaBlend(blended2, person, 255 - blend_mask)

    mask[np.where(gray_person > 10)] = 255
    mask = morph_trans(mask)
    mask = mask & mask1
    person[np.where(mask > 0)] = blended2[np.where(mask > 0)]
    return person

def reinit_vars():
    # 955, 1080
    global res_shape0, res_sape1, alpha, x_coords, y_corrds, glob_stage
    res_shape0 = 10
    res_sape1 = 10
    alpha = 0.4
    if glob_stage == 1:
        x_coords = 400
        y_corrds = 100
    elif glob_stage == 2:
        x_coords = 300
        y_corrds = 350


def draw_hands(opw, frame, person, angle1, angle2, angle3, angle4):
    global alpha, first_time, p3x, p3y, p4x, p4y, p6x, p6y, p7x, p7y
    pose_keypoints, image = opw.tag_person(frame)
    if first_time == 1:
        brat_dist1 = calculateDistance(pose_keypoints[0][2][0], pose_keypoints[0][2][1], pose_keypoints[0][3][0],
                                       pose_keypoints[0][3][1])
        p3x, p3y = calculatePoint(pose_keypoints[0][2][0], pose_keypoints[0][2][1], brat_dist1, angle1)
        antebrat_dist1 = calculateDistance(pose_keypoints[0][3][0], pose_keypoints[0][3][1], pose_keypoints[0][4][0],
                                           pose_keypoints[0][4][1])
        p4x, p4y = calculatePoint(p3x, p3y, antebrat_dist1, angle2)

        brat_dist2 = calculateDistance(pose_keypoints[0][5][0], pose_keypoints[0][5][1], pose_keypoints[0][6][0],
                                       pose_keypoints[0][6][1])
        p6x, p6y = calculatePoint(pose_keypoints[0][5][0], pose_keypoints[0][5][1], brat_dist2, angle3)
        antebrat_dist2 = calculateDistance(pose_keypoints[0][6][0], pose_keypoints[0][6][1], pose_keypoints[0][7][0],
                                           pose_keypoints[0][7][1])
        first_time = 0
        p7x, p7y = calculatePoint(p6x, p6y, antebrat_dist2, angle4)

    overlay = person.copy()
    d1 = calculateDistance(p3x, p3y, pose_keypoints[0][3][0],
                      pose_keypoints[0][3][1])
    d2 = calculateDistance(p4x, p4y, pose_keypoints[0][4][0],
                           pose_keypoints[0][4][1])
    d3 = calculateDistance(p6x, p6y, pose_keypoints[0][6][0],
                           pose_keypoints[0][6][1])
    d4 = calculateDistance(p7x, p7y, pose_keypoints[0][7][0],
                           pose_keypoints[0][7][1])
    if (d1 < 20 and d2 < 20 and d3 < 20 and d4 < 20):
        alpha = alpha + 0.02
    else:
        alpha = 0.4
    cv2.circle(overlay, (p3x, p3y), 35, circle_color, thickness=-1)
    cv2.circle(overlay, (p4x, p4y), 35, circle_color, thickness=-1)
    cv2.circle(overlay, (p6x, p6y), 35, circle_color, thickness=-1)
    cv2.circle(overlay, (p7x, p7y), 35, circle_color, thickness=-1)

    person_new = cv2.addWeighted(overlay, alpha, person, 1 - alpha, 0)
    showImage(person_new)

def draw_hands2(opw, frame, person, angle1, angle2, angle3, angle4, pose_keypoints):
    global alpha, first_time, p3x, p3y, p4x, p4y, p6x, p6y, p7x, p7y
    #pose_keypoints, image = opw.tag_person(frame)

    overlay = person.copy()
    p4x = 200
    p4y = 200

    p7x = 800
    p7y = 200

    cv2.circle(overlay, (p4x, p4y), 35, circle_color, thickness=-1)
    cv2.circle(overlay, (p7x, p7y), 35, circle_color, thickness=-1)


    d2 = calculateDistance(p4x, p4y, pose_keypoints[0][4][0],
                           pose_keypoints[0][4][1])
    d4 = calculateDistance(p7x, p7y, pose_keypoints[0][7][0],
                           pose_keypoints[0][7][1])
    if ( d2 < 60 and d4 < 60 and p4y < pose_keypoints[0][4][1] and p7y < pose_keypoints[0][7][1]):
        alpha = alpha + 0.1
    else:
        alpha = alpha - 0.1
        if alpha < 0.4:
            alpha = 0.4
    cv2.circle(overlay, (p4x, p4y), 35, circle_color, thickness=-1)
    cv2.circle(overlay, (p7x, p7y), 35, circle_color, thickness=-1)

    person_new = cv2.addWeighted(overlay, alpha, person, 1 - alpha, 0)
    return person_new



def pact(cap, frame, background, opw):
    global glob_stage
    start_time = time.time()
    checked_time = time.time()
    escaped = False
    i = 0
    pact_res = cv2.imread("ar/ress/pact.jpg")
    blood_res = cv2.imread("ar/ress/blood.jpg")

    black = np.zeros_like(background)
    dark_background = pact_res

    dark_background = cv2.resize(dark_background, (frame.shape[1], frame.shape[0]))
    dark_background = cv2.addWeighted(dark_background, 0.7, black, 0.3, 0)
    thick = 4
    blood = False
    while (not escaped):
        passed_time = time.time() - start_time
        frame = read_frame(cap)

        key, person = opw.tag_person(frame)

        gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame, dtype="uint8")
        mask[np.where(gray_person > 10)] = 255
        mask = 255 - mask
        person[np.where(mask > 0)] = dark_background[np.where(mask > 0)]

        if(passed_time > 3):

            d2 = calculateDistance(225, 425, key[0][4][0],
                                   key[0][4][1])
            if d2< 25:
                if thick < 30:
                    thick = thick + 2
                    checked_time = time.time()
                else:
                    blood = True
            cv2.rectangle(person, (200, 400), (250, 450), (255, 255, 255), thick, cv2.LINE_AA)
            if blood == True:
                person[395:455, 195:255] = blood_res
        if blood and (time.time() - checked_time > 3):
            escaped = True
        showImage(person)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i = i + 1
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    print(i)
    print("Passed time (s) in the first circle")
    print(passed_time)

