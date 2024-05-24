import os
import time
import uuid
import tempfile

import cv2
import gradio as gr
import importlib.util

import numpy as np

from tools import *

# init
# 临时文件
os.system(f"rm -r -f {TEMP_DIR}")
tempfile.tempdir = TEMP_DIR
os.makedirs(TEMP_DIR, exist_ok=TEMP_DIR)

# 人脸文件目录
os.makedirs(FACE_IMG_DIR, exist_ok=TEMP_DIR)


def import_module_by_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    modulevar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulevar)


def add_face(webcap_img, face_name):
    global FaceDB, FACE_IMG_DIR, INS_MODEL
    webcap_img = cv2.cvtColor(webcap_img, cv2.COLOR_RGB2BGR)

    if webcap_img is None or face_name is None:
        return "没有获取到图像", None

    face_name = face_name.strip()
    if len(face_name) == 0:
        return "没有获取到人员名称", None

    face_list = face_embedding(INS_MODEL, webcap_img, True)

    if len(face_list) == 0:
        return "没有检测到人脸", None

    face = face_list[0]

    img_name = uuid.uuid4()
    save_dir = os.path.join(FACE_IMG_DIR, face_name)
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(os.path.join(save_dir, f"{img_name}.jpg"), webcap_img)
    np.savez(os.path.join(save_dir, f"{img_name}.npy"), **face)

    if face_name in FaceDB.keys():
        FaceDB[face_name].append(face)
    else:
        FaceDB[face_name] = [face]
        FaceImgDB[face_name] = os.path.join(save_dir, f"{img_name}.jpg")


    show_img = webcap_img.copy()
    bbox = face['bbox']

    cv2.rectangle(show_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=5)

    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)

    return "录入成功", show_img


demo = gr.Blocks(title="人脸识别")
with demo:
    with gr.Row():
        gr.Markdown("# 人脸识别")

    for file in os.listdir(APP_DIR):
        if file.split(".")[-1] != "py":
            continue
        file_import_name = file.split(".")[0]
        with gr.Tab(file_import_name):
            import_module_by_path(file_import_name, os.path.join(os.getcwd(), APP_DIR, file))

    with gr.Row():
        gr.Markdown("### 增加人脸数据")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_face_name = gr.Textbox(label="人员名称", interactive=True)
            with gr.Row():
                face_add_input_img = gr.Image(label="通过摄像头拍摄", sources=['webcam', 'upload', 'clipboard'],
                                              interactive=True)

        with gr.Column():
            with gr.Row():
                add_output_img = gr.Image(label="录入人脸", interactive=False)
            with gr.Row():
                add_info = gr.Markdown()
            with gr.Row():
                add_face_button = gr.Button(value="添加人脸信息")
            with gr.Row():
                clear_face_button = gr.ClearButton(value="清空表单",
                                                   components=[face_add_input_img, input_face_name,
                                                               add_info, add_output_img])

    add_face_button.click(add_face,
                          [face_add_input_img, input_face_name],
                          [add_info, add_output_img])

if __name__ == '__main__':
    demo.launch(
        server_name="0.0.0.0",
        server_port=11000,
        # share=True
    )
