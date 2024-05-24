import os
import time

import cv2
import gradio as gr
import numpy as np
import face_recognition
from sklearn import preprocessing
from insightface.app import FaceAnalysis
from setting import *
from tools import *

det_app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'])


def get_face_info(input_img, threshold):
    global FaceDB, INS_MODEL, DEFAULT_THRESHOLD
    webcap_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    if webcap_img is None:
        return "没有获取到图像", None

    face_list, embedding_tyep = face_embedding(INS_MODEL, webcap_img, True)

    if len(face_list) == 0:
        if embedding_tyep:
            return "画面中人脸太小,人脸要求大小: w>100pix, h>100pix", None
        else:
            return "没有检测到人脸", None

    face = face_list[0]
    unknow_embedding = face['embedding'].reshape((1, -1))
    unknow_embedding = preprocessing.normalize(unknow_embedding)

    db_face_embedding_list = []
    db_face_name_list = []
    for face_name_info, face_data_list in FaceDB.items():
        for face_data in face_data_list:
            embedding = face_data['embedding'].reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            db_face_embedding_list.append(embedding.tolist()[0])
            db_face_name_list.append(face_name_info)

    db_face_embedding_list = np.array(db_face_embedding_list)
    dist_list = feature_compare(db_face_embedding_list, unknow_embedding)

    dist_argsort = np.argsort(dist_list)

    min_dist = dist_list[dist_argsort[0]]
    if min_dist < threshold:
        min_dist_name = db_face_name_list[dist_argsort[0]]
        show_img_path = FaceImgDB[min_dist_name]
        show_img = cv2.imread(show_img_path)
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        return show_img, f"检测得到人员: {min_dist_name}, 欧式距离: {min_dist}"

    else:
        return None, f"未知人员,最小距离 {min_dist}"


with gr.Row():
    with gr.Column():
        image_demo_input_image = gr.Image(label="输入图像", sources=['webcam', 'upload', 'clipboard'])
    with gr.Column():
        image_demo_output_image = gr.Image(label="匹配人脸", interactive=False)

with gr.Row():
    image_demo_threshold = gr.Slider(label="识别阈值", minimum=0.1, maximum=1, value=DEFAULT_THRESHOLD)

with gr.Row():
    image_demo_output_info = gr.Textbox(label="输出信息")

with gr.Row():
    with gr.Column():
        image_demo_input_button = gr.Button(value="上传")
    with gr.Column():
        image_demo_clear_button = gr.ClearButton(value="清空输入输出",
                                                 components=[image_demo_input_image,
                                                             image_demo_output_image,
                                                             image_demo_output_info])

    image_demo_input_button.click(
        get_face_info,
        [image_demo_input_image, image_demo_threshold],
        [image_demo_output_image, image_demo_output_info]
    )
