import cv2
import numpy as np
from tensorflow.keras.models import load_model
from face_detection import detect_faces
from tensorflow.keras.preprocessing import image

model = load_model('mask_detector_model.h5')

def predict_mask(frame, face):
    (x, y, w, h) = face
    face_region = frame[y:y+h, x:x+w]
    face_region = cv2.resize(face_region, (224, 224))
    img_array = image.img_to_array(face_region)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction
