from setting import *
import cv2
import math
import numpy as np


def det_face(det_model,img, ret_center=False):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(img.shape)

    det_res = det_model.get(img)

    if len(det_res) == 0:
        return []

    img_h, img_w = img.shape[:2]
    img_center_x = img_w / 2

    bbox_list = []
    center_d_list = []
    for face in det_res:
        x1, y1, x2, y2 = face['bbox']
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        bbox_list.append([x1, y1, x2, y2])

        box_c_x = (x1 + x2) / 2

        center_d_list.append(img_center_x - box_c_x)

    arg_sort = np.argsort(center_d_list)
    bbox_list_sort = [bbox_list[i] for i in arg_sort]

    if not ret_center:
        return bbox_list_sort
    else:
        center_sort = np.argsort([abs(i) for i in center_d_list])
        return bbox_list[center_sort[0]]


def cut_img(img, x, y, w, h, expand=1, border=True):
    if x is None or y is None or w is None or h is None:
        return img, [0, 0, img.shape[1], img.shape[0]]
    if w == 0 or h == 0:
        # return img, [0, 0, img.shape[1], img.shape[0]]
        output_img = img
        x1, y1 = 0, 0
        x2, y2 = img.shape[1], img.shape[0]
    else:
        cx = x + w / 2
        cy = y + h / 2

        x1 = cx - (w * expand) / 2
        y1 = cy - (h * expand) / 2
        x2 = x1 + w * expand
        y2 = y1 + h * expand

        x1 = math.ceil(max(0, x1))
        y1 = math.ceil(max(0, y1))
        x2 = math.ceil(min(img.shape[1], x2))
        y2 = math.ceil(min(img.shape[0], y2))
        output_img = img[y1:y2, x1:x2]

    if border:
        output_img = cv2.copyMakeBorder(output_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return output_img, [x1, y1, x2 - x1, y2 - y1]