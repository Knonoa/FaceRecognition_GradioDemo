import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis

# app = FaceAnalysis(name='buffalo_sc',allowed_modules=['detection'])  # 使用的检测模型名为buffalo_sc
app = FaceAnalysis(name='buffalo_l')  # 使用的检测模型名为buffalo_sc
print(app.models.keys())
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id小于0表示用cpu预测，det_size表示resize后的图片分辨率

img1 = cv2.imread("/mnt/f/Download/GuoDa1.jpg")  # 读取图片
img2 = cv2.imread("/mnt/f/Download/GuoDa2.jpg")  # 读取图片

time1 = time.time()
faces1 = app.get(img1)  # 得到人脸信息
faces2 = app.get(img2)
time2 = time.time()

print(faces2)

e1 = faces1[0]['embedding']
e2 = faces2[0]['embedding']


print(e1@e2.T)
print(np.sqrt(np.sum(np.square(e1-e2))))

print(time2-time1)



# rimg = app.draw_on(img, faces)   # 将人脸框绘制到图片上
# cv2.imwrite("/mnt/f/Download/GuoDa1_output.jpg", rimg)