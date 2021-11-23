import math
import cv2


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist
def calculatePoint(x1, y1, length, angle):
    P2x = (int)(x1 + length * math.cos(angle * math.pi / 180.0))
    P2y = (int)(y1 + length * math.sin(angle * math.pi / 180.0))
    return P2x, P2y