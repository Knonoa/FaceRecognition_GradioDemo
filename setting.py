import os
import face_recognition

FACE_IMG_DIR = "./FaceImage"
TEMP_DIR = "./TEMP"
APP_DIR = "./Apps"
DEFAULT_THRESHOLD = 0.5

FaceDB = {}
if os.path.exists(FACE_IMG_DIR):
    for face_name in os.listdir(FACE_IMG_DIR):
        for image_name in os.listdir(os.path.join(FACE_IMG_DIR, face_name)):
            picture = face_recognition.load_image_file(os.path.join(FACE_IMG_DIR, face_name, image_name))
            face_encoding = face_recognition.face_encodings(picture)[0]

            if face_name in FaceDB.keys():
                FaceDB[face_name].append(face_encoding)
            else:
                FaceDB[face_name] = [face_encoding]
