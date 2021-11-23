import cv2
import os
import datetime
import time
import numpy as np
import ctypes

from ar.utils import calculateDistance, calculatePoint
from openpose.open_pose_extractorv2 import OpenPoseWrapper
from person_extractor.person_extractor import person_extraction2
from utils import showImage, read_frame

background_file = '../doc/background_8.jpg'
w_width = 1916
w_height = 1080

x_coords = 100
y_corrds = 600
alpha = 0.4
first_time = 1
p3x, p3y, p4x, p4y, p6x, p6y, p7x, p7y = np.zeros(8)
circle_color = (255, 255, 255)
res_shape0, res_shape1 = (0, 0)
glob_stage = 0
start_video = False
stop_video = False

def morph_trans(mask):
    dilatation_size = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    mask = cv2.erode(mask, element)
    mask = cv2.dilate(mask, element)
    '''dilatation_size = 10
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    mask = cv2.dilate(mask, element)'''
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
    mask1 = mask.copy()
    layer2 = mask.copy()
    blend_mask = mask.copy()

    layer2[x_coords: x_coords + res_shape0, y_corrds: y_corrds + res_shape1] = res[:res_shape0, :res_shape1]
    #cv2.imshow('layer2', layer2)
    mask1[x_coords: x_coords + res_shape0, y_corrds: y_corrds + res_shape1] = 255
    #cv2.imshow('mask1', mask1)

    cv2.circle(blend_mask, (  int((2 * y_corrds + res_shape1) / 2), int((2 * x_coords + res_shape0) / 2)),
               int(max(res_shape0, res_shape1) / 2)
               , (255, 255, 255), -1, cv2.LINE_AA)
    #cv2.imshow('blend_mask1', blend_mask)
    blend_mask = cv2.GaussianBlur(blend_mask, (121, 121), 11)
    #cv2.imshow('blend_mask2', blend_mask)
    blured = cv2.GaussianBlur(layer2, (21, 21), 11)
    #cv2.imshow('bblured', blured)
    blended2 = alphaBlend(layer2, blured, 255 - blend_mask)
    #cv2.imshow('blend2', blended2)
    blended2 = alphaBlend(blended2, person, 255 - blend_mask)
    #cv2.imshow('blend3', blended2)

    mask = cv2.inRange(gray_person, 10, 255)
    dilatation_size = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    #cv2.imshow('mask_before_erode', mask)
    mask = cv2.erode(mask, element)
    #cv2.imshow('mask_after_erode', mask)
    mask = cv2.dilate(mask, element)
    #cv2.imshow('mask_dilate', mask)
    mask = mask & mask1[:, :, 0]
    #cv2.imshow('mask_fin', mask)

    person[np.where(mask > 0)] = blended2[np.where(mask > 0)]
    #cv2.imshow('person', person)
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
    elif glob_stage == 10:
        x_coords = 100
        y_corrds = 200
    elif glob_stage == 11:
        x_coords = 200
        y_corrds = 600
    elif glob_stage == 12:
        x_coords = 500
        y_corrds = 100
    elif glob_stage == 20:
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

def circle1(cap, background, ring, ring_mask, opw, background_ress):
    global glob_stage
    start_time = time.time()
    escaped = False
    i = 0
    isis_res = cv2.imread("ar/ress/isis-2.png")
    isis_res = cv2.flip(isis_res, 1)
    putin_res = cv2.imread("ar/ress/hitler.jpg")
    putin_res = cv2.flip(putin_res, 1)
    blm_res = cv2.imread("ar/ress/blm.png")
    blm_res = cv2.flip(blm_res, 1)

    black = np.zeros_like(background)
    black[:, :] = (10,0,0)
    dark_background = cv2.addWeighted(background_ress, 0.05, black, 0.95, 0)
    #dark_ring
    dark_ring = cv2.addWeighted(ring, 0.6, black, 0.4, 0, ring_mask)

    while (not escaped):
        passed_time = time.time() - start_time
        frame = read_frame(cap)
        key, person = opw.tag_person(frame)
        person2 = person_extraction2(frame, background)
        cap.save_train(person, person2, key)

        gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        gray_person2 = cv2.cvtColor(person2, cv2.COLOR_BGR2GRAY)

        mask = ring_mask.copy()
        mask[np.where(gray_person > 10)] = 0
        person[np.where(mask > 0)] = dark_ring[np.where(mask > 0)]
        mask[np.where(gray_person > 10)] = 255
        mask = 255 - mask
        person[np.where(mask > 0)] = dark_background[np.where(mask > 0)]

        stage = 0
        if ((passed_time>5 and passed_time<15) or(passed_time>40 and passed_time < 50)
            or (passed_time > 75 and passed_time < 85)):
            stage = 0

        elif((passed_time>=15 and passed_time<25) or(passed_time>50 and passed_time < 60) or
            (passed_time > 85 and passed_time < 95)):
            stage = 1
        elif ((passed_time >= 25 and passed_time < 35) or (passed_time > 60 and passed_time < 70) or
                  (passed_time > 95 and passed_time < 105)):
            stage = 2

        if (passed_time > 5 and passed_time < 35):
            person = add_layer(isis_res, person, frame, gray_person2, stage)
            glob_stage = 1
        elif (passed_time > 40 and passed_time < 70):
            person = add_layer(putin_res, person, frame, gray_person2, stage)
            glob_stage = 2
        elif (passed_time > 75 and passed_time < 105):
            person = add_layer(blm_res, person, frame, gray_person2, stage)
        elif (passed_time > 105):
            if (alpha > 0.6):
                escaped = True
            person = draw_hands2(opw, frame, person, angle1=180, angle2=90, angle3=0, angle4=90, pose_keypoints=key)
        else:
            reinit_vars()
        showImage(person)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i = i + 1
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    print(i)
    print("Passed time (s) in the first circle")
    print(passed_time)


def circle2(cap, background, ring, ring_mask, opw, background_ress):
    global glob_stage
    glob_stage = 10
    start_time = time.time()
    escaped = False;
    i = 0

    beauty_res = cv2.imread("ar/ress/beauty.jpg")
    beauty_res = cv2.flip(beauty_res, 1)
    kylei_res = cv2.imread("ar/ress/kiley2.jpg")
    kylei_res = cv2.flip(kylei_res, 1)
    beauty2_res = cv2.imread("ar/ress/beauty2.jpg")
    beauty2_res = cv2.flip(beauty2_res, 1)

    black = np.zeros_like(background)
    black[:, :] = (0, 0, 8)
    dark_background = cv2.addWeighted(background_ress, 0.05, black, 0.95, 0)
    # dark_ring
    dark_ring = cv2.addWeighted(ring, 0.6, black, 0.4, 0, ring_mask)

    while (not escaped):
        passed_time = time.time() - start_time

        frame = read_frame(cap)
        #print(i)
        key, person = opw.tag_person(frame)
        #e1 = cv2.getTickCount()
        person2 = person_extraction2(frame, background)
        #e2 = cv2.getTickCount()
        #timee = (e2 - e1) / cv2.getTickFrequency()
        #print("person extraction ")
        #print(timee)
        gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        gray_person2 = cv2.cvtColor(person2, cv2.COLOR_BGR2GRAY)

        #e1 = cv2.getTickCount()
        mask = ring_mask.copy()
        mask[np.where(gray_person > 10)] = 0
        person[np.where(mask > 0)] = dark_ring[np.where(mask > 0)]
        mask[np.where(gray_person > 10)] = 255
        mask = 255 - mask
        person[np.where(mask > 0)] = dark_background[np.where(mask > 0)]
        #e2 = cv2.getTickCount()
        #timee = (e2 - e1) / cv2.getTickFrequency()
        #print("matrix ")
        #print(timee)

        stage = 0
        if ((passed_time>5 and passed_time<15) or(passed_time>40 and passed_time < 50)
            or (passed_time > 75 and passed_time < 85)):
            stage = 0

        elif((passed_time>=15 and passed_time<25) or(passed_time>50 and passed_time < 60) or
            (passed_time > 85 and passed_time < 95)):
            stage = 1
        elif ((passed_time > 25 and passed_time < 35) or (passed_time > 60 and passed_time < 70) or
                  (passed_time > 95 and passed_time < 105)):
            stage = 2


        if (passed_time > 5 and passed_time < 35):
            #e1 = cv2.getTickCount()
            person = add_layer(kylei_res, person, frame, gray_person2, stage)
            #e2 = cv2.getTickCount()
            #timee = (e2 - e1) / cv2.getTickFrequency()
            #print("add layer ")
            #print(timee)
            glob_stage = 11
        elif (passed_time > 40 and passed_time < 70):
            person = add_layer(beauty_res, person, frame, gray_person2, stage)
            glob_stage = 12
        elif (passed_time > 75 and passed_time < 105):
            person = add_layer(beauty2_res, person, frame, gray_person2, stage)
        elif (passed_time > 107):
            if (alpha > 0.6):
                escaped = True
            person = draw_hands2(opw, frame, person, angle1=180, angle2=90, angle3=0, angle4=90, pose_keypoints=key)
        else:
            reinit_vars()
        showImage(person)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i = i + 1
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    print(i)
    print("Passed time (s) in the first circle")
    print(passed_time)


def add_video_layer(blm_cap, blm_frame, person, frame, gray_person):
    global res_shape0, res_shape1, start_video, stop_video

    if (start_video):
        ret, blm_frame = blm_cap.read()
        if not ret:
            stop_video = True
            return person
    res_shape1 = blm_frame.shape[1]
    res_shape0 = blm_frame.shape[0]

    mask = np.zeros_like(frame)
    mask1 = mask.copy()
    layer2 = mask.copy()
    blend_mask = mask.copy()
    layer2[x_coords: x_coords + res_shape0, y_corrds: y_corrds + res_shape1] = blm_frame[:res_shape0, :res_shape1]
    mask1[x_coords: x_coords + res_shape0, y_corrds: y_corrds + res_shape1] = 255

    cv2.circle(blend_mask, (  int((2 * y_corrds + res_shape1) / 2), int((2 * x_coords + res_shape0) / 2)),
               int(max(res_shape0, res_shape1) / 2)
               , (255, 255, 255), -1, cv2.LINE_AA)
    blend_mask = cv2.GaussianBlur(blend_mask, (121, 121), 11)
    blured = cv2.GaussianBlur(layer2, (21, 21), 11)
    blended2 = alphaBlend(layer2, blured, 255 - blend_mask)
    blended2 = alphaBlend(blended2, person, 255 - blend_mask)
    mask = cv2.inRange(gray_person, 10, 255)
    mask = morph_trans(mask)
    mask = mask & mask1[:, :, 0]
    person[np.where(mask > 0)] = blended2[np.where(mask > 0)]
    if not start_video:
        if(np.count_nonzero(mask)) > 0:
            start_video = True
    return person

def circle3(cap, background, ring, ring_mask, opw, background_ress):
    global glob_stage
    glob_stage = 20
    start_time = time.time()
    escaped = False;
    i = 0
    blm_cap = cv2.VideoCapture("ar/ress/blm2.mp4")
    ret, blm_frame = blm_cap.read()
    black = np.zeros_like(background)
    dark_background = cv2.addWeighted(background_ress, 0.05, black, 0.95, 0)
    dark_ring = cv2.addWeighted(ring, 0.6, black, 0.4, 0, ring_mask)

    while (not escaped):
        passed_time = time.time() - start_time

        frame = read_frame(cap)
        key, person = opw.tag_person(frame)
        person2 = person_extraction2(frame, background)
        gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        gray_person2 = cv2.cvtColor(person2, cv2.COLOR_BGR2GRAY)
        mask = ring_mask.copy()
        mask[np.where(gray_person > 10)] = 0
        person[np.where(mask > 0)] = dark_ring[np.where(mask > 0)]
        mask[np.where(gray_person > 10)] = 255
        mask = 255 - mask
        person[np.where(mask > 0)] = dark_background[np.where(mask > 0)]

        if (passed_time > 5 and not stop_video):
            person = add_video_layer(blm_cap, blm_frame, person, frame, gray_person2)
        if (stop_video):
            if (alpha > 0.6):
                escaped = True
            person = draw_hands2(opw, frame, person, angle1=180, angle2=90, angle3=0, angle4=90, pose_keypoints=key)
        else:
            reinit_vars()
        showImage(person)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i = i + 1
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    print(i)
    print("Passed time (s) in the first circle")
    print(passed_time)
