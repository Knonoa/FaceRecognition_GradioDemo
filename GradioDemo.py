import os
import time
import uuid
import shutil
import tempfile

import cv2
import gradio as gr
import importlib.util
import face_recognition
from setting import *

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
    global FaceDB, FACE_IMG_DIR

    face_name = face_name.strip()

    if webcap_img is None:
        return "错误! 没有收到任何文件"

    if len(face_name.strip()) == 0:
        return "错误! 必须填写人员名称"

    start_time = time.time()
    face_encodings = face_recognition.face_encodings(webcap_img)
    if len(face_encodings) > 1:
        return f"错误! 照片种存在多个({len(face_encodings)})人脸,不支持多人脸输入"
    if len(face_encodings) == 0:
        return f"错误! 照片种没有人脸"

    face_encodings = face_encodings[0]

    # 拷贝文件到指定目录
    save_dir = os.path.join(FACE_IMG_DIR, face_name)
    os.makedirs(save_dir, exist_ok=True)
    new_name = f"{uuid.uuid4()}.jpg"
    cv2.imwrite(os.path.join(save_dir, new_name), cv2.cvtColor(webcap_img, cv2.COLOR_BGR2RGB))

    end_time = time.time()

    if face_name not in FaceDB.keys():
        FaceDB[face_name] = [face_encodings]
        return f"新增人脸数据: {face_name}\n耗时:{end_time - start_time}s"
    else:
        FaceDB[face_name].append(face_encodings)
        return f"人脸数据已存在, 新增人脸数据: {face_name}, 当前人员有{len(FaceDB[face_name])}条数据\n耗时:{end_time - start_time}s"


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
                face_add_input_img = gr.Image(label="通过摄像头拍摄", sources=['webcam', 'upload', 'clipboard'], interactive=True)

        with gr.Column():
            with gr.Row():
                add_info = gr.Markdown()
            with gr.Row():
                add_face_button = gr.Button(value="添加人脸信息")
            with gr.Row():
                clear_face_button = gr.ClearButton(value="清空表单",
                                                   components=[face_add_input_img, input_face_name,
                                                               add_info])

    add_face_button.click(add_face,
                          [face_add_input_img, input_face_name],
                          [add_info])

if __name__ == '__main__':
    demo.launch(
        server_name="0.0.0.0",
        server_port=11000,
        share=True
    )