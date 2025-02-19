import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model("face_mask_detection/models/mask_detector.keras")

# Define test data path
test_dir = "/home/nyxus/Documents/ML_Engineer/face_mask_detection/dataset/test "  # Adjust path if needed

# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")
