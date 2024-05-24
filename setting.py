import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

FACE_IMG_DIR = "./FaceImage"
TEMP_DIR = "./TEMP"
APP_DIR = "./Apps"
DEFAULT_THRESHOLD = 0.8

INS_MODEL = FaceAnalysis(name='buffalo_l')
INS_MODEL.prepare(ctx_id=-1, det_size=(640, 640))

FaceDB = {}
FaceImgDB = {}
if os.path.exists(FACE_IMG_DIR):
    for face_name in os.listdir(FACE_IMG_DIR):
        face_name_dir = os.path.join(FACE_IMG_DIR, face_name)
        for file_name in os.listdir(face_name_dir):
            if ".npz" not in file_name:
                continue

            with np.load(os.path.join(face_name_dir, file_name)) as data:
                dict_loaded = dict(data.items())

            if face_name in FaceDB.keys():
                FaceDB[face_name].append(dict_loaded)
            else:
                FaceDB[face_name] = [dict_loaded]

if os.path.exists(FACE_IMG_DIR):
    for face_name in os.listdir(FACE_IMG_DIR):
        face_name_dir = os.path.join(FACE_IMG_DIR, face_name)
        for file_name in os.listdir(face_name_dir):
            if ".jpg" in file_name:
                FaceImgDB[face_name] = os.path.join(face_name_dir, file_name)
                break
