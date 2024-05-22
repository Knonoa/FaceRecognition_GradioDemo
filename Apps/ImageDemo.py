import os
import time

import cv2
import gradio as gr
import numpy as np
import face_recognition
from setting import *


def get_face_info(input_img,threshold):
    global FaceDB, FACE_IMG_DIR

    if input_img is None:
        return None, "没有图片输入"

    start_time = time.time()
    face_encoding = face_recognition.face_encodings(input_img)
    if len(face_encoding) > 1:
        return None, "存在多张人脸"
    if len(face_encoding) == 0:
        return None, "没有检测到人脸"

    all_face_embedding = []
    all_face_name = []
    for face_name, face_list in FaceDB.items():
        for face in face_list:
            all_face_embedding.append(face)
            all_face_name.append(face_name)

    scores = face_recognition.face_distance(all_face_embedding, face_encoding[0])
    scores_arg = np.argsort(scores)[0]
    if scores[scores_arg] > threshold:
        return None, "未知人员"

    else:
        output_face_name = all_face_name[scores_arg]
        output_score = scores[scores_arg]

        face_name_dir = os.path.join(FACE_IMG_DIR, output_face_name)
        show_img_path = os.path.join(face_name_dir, os.listdir(face_name_dir)[0])
        show_img = cv2.imread(show_img_path)
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)

        return show_img, f"检测到人员: {output_face_name}, 欧式距离: {output_score}, 识别阈值: {threshold}"


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
        [image_demo_input_image,image_demo_threshold],
        [image_demo_output_image, image_demo_output_info]
    )