import tensorflow as tf
import cv2
import os
import time
import numpy as np

from ar.animation import alphaBlend
from ar.utils import calculateDistance
from gan.HandGan import normalize, resize
from gan.Loader import Loader
from openpose.open_pose_extractorv2 import OpenPoseWrapper
from part_extractor.part_extractor import get_body
from person_extractor.person_extractor import person_extraction2
from utils import read_frame, showImage
import glob

dataset_folder = "../dataset"

def gen_gan(person, loader, frame):
    input_image = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    real_image = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    input_image, real_image = resize(input_image, real_image, 256, 256)
    input_image, real_image = normalize(input_image, real_image)

    input_image = tf.expand_dims(input_image, axis=0)
    real_image = tf.expand_dims(real_image, axis=0)

    prediction = loader.generate_images(input_image, real_image)

    res = cv2.cvtColor(np.array(prediction), cv2.COLOR_RGB2BGR)
    res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    res = cv2.resize(res, (frame.shape[1], frame.shape[0]))
    return res

def profile(cap, frame, background, opw):
    i = 0
    index = 0
    start_time = time.time()
    black = np.zeros_like(background)
    black[:, :] = (15, 0, 0)
    background2 = cv2.imread("gan/prof.jpg")
    background2 = cv2.resize(background2, (frame.shape[1], frame.shape[0]))
    dark_background = cv2.addWeighted(background, 0.05, black, 0.95, 0)
    loader = Loader()
    filenames = glob.glob("motions/dance/*.jpg")
    filenames.sort()
    images = [{'img': cv2.imread(img), 'name': img} for img in filenames]

    filenames = glob.glob("motions/wave_left/*.jpg")
    filenames.sort()
    wave_left = [{'img': cv2.imread(img), 'name': img} for img in filenames]

    filenames = glob.glob("motions/wave_right/*.jpg")
    filenames.sort()
    wave_right = [{'img': cv2.imread(img), 'name': img} for img in filenames]

    ll = len(images)
    l2 = len(wave_right)
    l3 = len(wave_left)
    idle = True
    left = False
    right = False
    dance = False
    k = 0
    # while (True):
    #for filename in os.listdir(dataset_folder):
    while (True):
        passed_time = time.time() - start_time
        frame = read_frame(cap)
        #frame = cv2.imread(os.path.join(dataset_folder, filename))
        if passed_time > 800 and idle:
            idle = False
            if k % 3 == 0:
                left = True
            if k % 3 == 1:
                right = True
            if k % 3 == 2:
                dance = True
        if (idle):
            key, person = opw.tag_person(frame)
            person2 = person_extraction2(frame, background)
            head, head_mask = get_body(key, person2)

            person = gen_gan(person, loader, frame)
            gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(gray_person, 10, 255)
            if head_mask is None:
                showImage(dark_background)
                continue
            inv = 255 - head_mask
            inv = cv2.bitwise_and(inv, inv, mask=head_mask[:, :, 0])
            head = head / 255.0
            inv = inv / 255.0
            res_head = np.uint8(head * head_mask + person * inv)

            head_mask = head_mask[:, :, 0]
            head_mask = cv2.inRange(head_mask, 1, 255)
            person = cv2.bitwise_and(person, person, mask=255 - head_mask)
            res_head2 = cv2.bitwise_and(res_head, res_head, mask=head_mask)

            p0_x = key[0][0][0]
            p0_y = key[0][0][1]
            p1_x = key[0][1][0]
            p1_y = key[0][1][1]
            p2_x = key[0][16][0]
            p2_y = key[0][16][1]
            p3_x = key[0][17][0]
            p3_y = key[0][17][1]
            center_x = int((p2_x + p3_x) / 2)
            center_y = int((p2_y + p3_y) / 2) - 30
            horizontal = int(calculateDistance(p2_x, p2_y, p3_x, p3_y))
            vertical = int(calculateDistance(p0_x, p0_y, p1_x, p1_y))
            axesLength = (int(horizontal * 1.5), int(vertical * 1.2))

            mask_gan = np.zeros_like(frame)
            mask_gan = cv2.ellipse(mask_gan, (center_x, center_y), axesLength, 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
            mask_gan = 255 - mask_gan

            person = cv2.bitwise_and(person, person, mask=mask_gan[:, :, 0])

            person = cv2.bitwise_or(person, res_head2)

            #person[np.where(head_mask > 0)] = res_head[np.where(head_mask > 0)]
            gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            # mask[np.where(gray_person > 10)] = 255
            mask = cv2.inRange(gray_person, 10, 255)
            person = cv2.bitwise_and(person, person, mask = mask)
            background2 = cv2.bitwise_and(dark_background, dark_background, mask = 255-mask)
            person = cv2.bitwise_or(person, background2)
            #person[np.where(mask < 10)] = dark_background[np.where(mask < 10)]

            '''e2 = cv2.getTickCount()
            timee = (e2 - e1) / cv2.getTickFrequency()
            print("rest ")
            print(timee)'''

            showImage(person)
        else:
            print(i)
            key, person = opw.tag_person(frame)


            head, head_mask = get_body(key, person2)
            if dance:
                name = images[index]['name']
                name = name[:-4]
                name = name.split("__")
                my = int(name[1]) - 50
                mx = int(name[2]) + 20
                person = gen_gan(images[index]['img'], loader, frame)
            elif right:
                name = wave_right[index]['name']
                name = name[:-4]
                name = name.split("__")
                my = int(name[1]) - 50
                mx = int(name[2]) + 20
                person = gen_gan(wave_right[index]['img'], loader, frame)
            elif left:
                name = wave_left[index]['name']
                name = name[:-4]
                name = name.split("__")
                my = int(name[1]) - 50
                mx = int(name[2]) + 20
                person = gen_gan(wave_left[index]['img'], loader, frame)
            #e1 = cv2.getTickCount()
            try:
                gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
                #mask = np.zeros_like(frame, dtype=np.uint8)
                #mask[np.where(gray_person > 10)] = 255
                mask = cv2.inRange(gray_person, 10, 255)
                inv = 255 - head_mask
                inv = cv2.bitwise_and(inv, inv, mask=head_mask[:, :, 0])

                head_mask2 = np.zeros_like(frame, dtype=np.uint8)
                head2 =np.zeros_like(frame, dtype=np.uint8)
                inv2 = np.zeros_like(frame, dtype=np.uint8)
                #cv2.circle(person, (mx, my), 30, (255, 255, 255), -1, cv2.LINE_AA)

                minx = min(np.where(head > 2)[0])
                miny = min(np.where(head > 2)[1])
                maxx = max(np.where(head > 2)[0])
                maxy = max(np.where(head > 2)[1])
                dxf = (maxx - minx) / 2
                dx = int(dxf)

                dyf = (maxy - miny) / 2
                dy = int(dyf)


                end_x = mx + dx
                end_y = my + dy

                if (end_x > frame.shape[0]):
                    maxx = maxx - (end_x - frame.shape[0])
                    end_x = frame.shape[0]
                if (end_y > frame.shape[1]):
                    maxy = maxy - (end_y - frame.shape[1])
                    end_y = frame.shape[1]


                head_mask2_shape = head_mask2[mx - dx:end_x,my - dy:end_y].shape
                head_mask_shape = head_mask[minx: maxx, miny:maxy].shape

                if (head_mask2_shape[0] == head_mask_shape[0] + 1):
                    maxx = maxx + 1
                if (head_mask2_shape[1] == head_mask_shape[1] + 1):
                    maxy = maxy + 1
                if (head_mask2_shape[0] == head_mask_shape[0] - 1):
                    maxx = maxx - 1
                if (head_mask2_shape[1] == head_mask_shape[1] - 1):
                    maxy = maxy - 1

                head2[mx - dx:end_x, my - dy:end_y] = head[minx: maxx, miny:maxy]
                head_mask2[mx - dx:end_x,my - dy:end_y] = head_mask[minx: maxx, miny:maxy]
                inv2[mx - dx:end_x,my - dy:end_y] = inv[minx: maxx, miny:maxy]

                head2 = head2 / 255.0
                inv2 = inv2 / 255.0
                res_head = np.uint8(head2 * head_mask2 + person * inv2)

                head_mask2 = head_mask2[:, :, 0]
                head_mask2 = cv2.inRange(head_mask2, 1, 255)
                person = cv2.bitwise_and(person, person, mask=255 - head_mask2)
                res_head2 = cv2.bitwise_and(res_head, res_head, mask=head_mask2)
                person = cv2.bitwise_or(person, res_head2)
                #person[np.where(head_mask2 > 0)] = res_head[np.where(head_mask2 > 0)]

                person = cv2.bitwise_and(person, person, mask=mask)
                background2 = cv2.bitwise_and(dark_background, dark_background, mask=255 - mask)
                person = cv2.bitwise_or(person, background2)
                #person[np.where(mask < 10)] = dark_background[np.where(mask < 10)]

                '''e2 = cv2.getTickCount()
                timee = (e2 - e1) / cv2.getTickFrequency()
                print("rest ")
                print(timee)'''
                showImage(person)
                index = index + 1
                if (index >= ll and dance):
                    start_time = time.time()
                    idle = True
                    index = 0
                    k = k + 1
                if (index >= l2 and right):
                    start_time = time.time()
                    idle = True
                    index = 0
                    k = k + 1
                if (index >= l3 and left):
                    start_time = time.time()
                    idle = True
                    index = 0
                    k = k + 1
            except:
                print()

        '''except Exception as e:
            print(e)
            showImage(dark_background)'''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i = i + 1
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    print(i)
    print("Passed time (s) in the profile")
    print(passed_time)
