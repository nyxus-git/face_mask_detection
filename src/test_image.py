import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("../models/mask_detector.keras")

# Load and preprocess image
image_path = "../dataset/test/mask/sample.jpg"  # Change path
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224)) 
image = np.expand_dims(image, axis=0) / 255.0  

# Make prediction
prediction = model.predict(image)[0][0]

# Output result
if prediction > 0.5:
    print("❌ No Mask Detected!")
else:
    print("✅ Mask Detected!")
