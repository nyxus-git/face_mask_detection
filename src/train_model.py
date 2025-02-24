import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Paths
dataset_path = "dataset/train"  # Ensure your dataset folder is inside the project
categories = ["with_mask", "without_mask"]

# Data & Labels
data, labels = [], []

# Load and Preprocess Images
print("📂 Loading dataset...")

for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 128))
        image = np.array(image, dtype="float32") / 255.0  # Normalize
        
        data.append(image)
        labels.append(label)

# Convert to NumPy Arrays
data = np.array(data)
labels = np.array(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert Labels to Categorical
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

print("✅ Dataset Loaded & Preprocessed!")

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile Model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
print("🚀 Training Model...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test),class_weight={0: 1.5, 1: 1.0})

# Save Model
model.save("models/mask_detector.keras")  # Recommended
print("✅ Model Training Completed & Saved!")
