import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis


def face_embedding(det_model, img, ret_center=False):
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

        box_c_x = x1 + (x2 - x1) / 2

        center_d_list.append(abs(img_center_x - box_c_x))

        print(x2 - x1, y2 - y1,box_c_x,img_center_x,box_c_x - img_center_x)

    arg_sort = np.argsort(center_d_list)
    output = [det_res[i] for i in arg_sort]

    if not ret_center:
        return output
    else:
        return [output[arg_sort[0]]]


# app = FaceAnalysis(name='buffalo_sc',allowed_modules=['detection'])  # 使用的检测模型名为buffalo_sc
app = FaceAnalysis(name='buffalo_l')  # 使用的检测模型名为buffalo_sc
print(app.models.keys())
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id小于0表示用cpu预测，det_size表示resize后的图片分辨率

img1 = cv2.imread("0d45f625-cea8-428a-a88f-1ba7c00e3f4d.png")  # 读取图片

face_list = face_embedding(app, img1, False)

for face_i, face in enumerate(face_list):
    output_img = img1.copy()
    bbox = face['bbox']
    cv2.rectangle(output_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0),
                  thickness=5)
    cv2.imwrite(f'test{face_i}.jpg', output_img)
