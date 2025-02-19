import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces TensorFlow to use CPU

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained mask detection model
MODEL_PATH = "./models/mask_detector.keras"  # Adjust path if needed
model = load_model(MODEL_PATH)

# Ensure model input shape is correct
expected_input_shape = model.input_shape
print("Model Expected Input Shape:", expected_input_shape)

# Load OpenCV's pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam!")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face = cv2.resize(face, (128, 128))  # Resize to model input size
        face = img_to_array(face) / 255.0  # Normalize
        face = np.expand_dims(face, axis=0)
        
        # Reshape input if necessary (flatten if required by model)
        if expected_input_shape[-1] == 57600:
            face = face.reshape((1, -1))

        # Predict mask or no mask
        (mask, withoutMask) = model.predict(face)[0]

        # Determine label and color for bounding box
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Display label and rectangle
        cv2.putText(frame, f"{label}: {max(mask, withoutMask) * 100:.2f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show the video feed
    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
