import math
import cv2
import numpy as np

from ar.utils import calculateDistance
length = 30
length_hand = 50


def get_perpendicular(p2_x, p2_y, p3_x, p3_y):
    vx = p2_x - p3_x
    vy = p2_y - p3_y
    mag = math.sqrt(vx * vx + vy * vy)
    vx = vx / mag
    vy = vy / mag
    temp = vx
    vx = -vy
    vy = temp
    return vx, vy


def get_bounding_points(p2_x, p2_y, vx, vy):
    global length
    cx2 = p2_x - vx * length
    cy2 = p2_y - vy * length

    cx = p2_x + vx * length
    cy = p2_y + vy * length
    return cx2, cy2, cx, cy


def get_upper_rect(p2_x, p2_y, p3_x, p3_y):
    vx, vy = get_perpendicular(p2_x, p2_y, p3_x, p3_y)
    l1x, l1y, l2x, l2y = get_bounding_points(p2_x, p2_y, vx, vy)
    l3x, l3y, l4x, l4y = get_bounding_points(p3_x, p3_y, vx, vy)
    return int(l1x), int(l1y), int(l2x), int(l2y), int(l3x), int(l3y), int(l4x), int(l4y)


def get_lower_rect(p3_x, p3_y, p4_x, p4_y):
    vx, vy = get_perpendicular(p3_x, p3_y, p4_x, p4_y)
    l1x, l1y, l2x, l2y = get_bounding_points(p4_x, p4_y, vx, vy)
    l3x, l3y, l4x, l4y = get_bounding_points(p3_x, p3_y, vx, vy)
    return int(l3x), int(l3y), int(l4x), int(l4y), int(l1x), int(l1y), int(l2x), int(l2y)


def get_palm(p3x, p3y, p4x, p4y):
    vx = p3x - p4x
    vy = p3y - p4y
    mag = math.sqrt(vx * vx + vy * vy)
    vx = vx / mag
    vy = vy / mag
    cx = p4x - length_hand * vx;
    cy = p4y - length_hand * vy
    pvx, pvy = get_perpendicular(p3x, p3y, p4x, p4y)
    l1x, l1y, l2x, l2y = get_bounding_points(cx, cy, pvx, pvy)
    return int(l1x), int(l1y), int(l2x), int(l2y)


def get_hand(keypoints, image, frame):
    for i in keypoints:
        if (i[2][2] > 0.3):
            p2_x = i[2][0]
            p2_y = i[2][1]
            p3_x = i[3][0]
            p3_y = i[3][1]
            p4_x = i[4][0]
            p4_y = i[4][1]
            break;
    l1x, l1y, l2x, l2y, l3x, l3y, l4x, l4y = get_upper_rect(p2_x, p2_y, p3_x, p3_y)
    l5x, l5y, l6x, l6y, l7x, l7y, l8x, l8y = get_lower_rect(p3_x, p3_y, p4_x, p4_y)

    l9x, l9y, l10x, l10y = get_palm(p3_x, p3_y, p4_x, p4_y)

    pts = np.array([[l1x, l1y], [l2x, l2y], [l4x, l4y], [l6x, l6y], [l8x, l8y], [l10x, l10y], [l9x, l9y],
                    [l7x, l7y], [l5x, l5y], [l3x, l3y]])
    pts = pts.reshape((-1, 1, 2))
    hull = cv2.convexHull(pts)
    # image = cv2.polylines(image, [hull], True, (0, 255, 0), thickness=2)

    height = frame.shape[0]
    width = frame.shape[1]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], (255))
    res = cv2.bitwise_and(frame, frame, mask=mask)

    inv_mask = 255 - mask
    # inv_mask[int(p2_x) - 100: int(p2_x) + 200, int(p2_y) - 100: int(p2_y) + 100] = 255

    rest_of_body = cv2.bitwise_and(frame, frame, mask=inv_mask)
    return res, rest_of_body


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


def get_body(keypoints, frame):
    try:
        if not keypoints is None:
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
                    horizontal = int(calculateDistance(p2_x, p2_y, p3_x, p3_y))
                    vertical = int(calculateDistance(p0_x, p0_y, p1_x, p1_y))
                    axesLength = (int(horizontal), int(2 * vertical))
                    mask = cv2.ellipse(mask, (center_x, center_y),axesLength, 0, 0, 360, (255, 255, 255),  -1, cv2.LINE_AA)
                    head = cv2.bitwise_and(frame, frame, mask=mask)

                    gray_person = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)

                    head_mask = cv2.inRange(gray_person, 10, 255)
                    head_mask = cv2.merge((head_mask, head_mask, head_mask))

                    array = get_gradient_3d(3 * horizontal, 2 * vertical,  (255, 255, 255), (0, 0, 0), (False, False, False))
                    array = array.astype(np.uint8)
                    head_mask[ center_y  + 10 : center_y +  2 * vertical + 10 , center_x - horizontal : center_x + horizontal] = array[:,  :2 * horizontal]

                    head_mask = cv2.bitwise_and(head_mask, head_mask, mask=gray_person)
                    #head_mask = cv2.bitwise_and(head_mask, head_mask, mask=frame[:, :, 0])
                    return head, head_mask
    except Exception:
            pass
    return None, None