import os
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import joblib
import tensorflow as tf  # Add this line

# Define paths
data_dir = "C:/Users/shakt/Downloads/Data"  # Adjust path if needed

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32

# Create an instance of ImageDataGenerator
datagen = ImageDataGenerator(validation_split=0.2)  # Add validation split if needed


# Custom function to load a random limited number of images
def random_sample_images(directory, class_indices, max_images_per_class):
    data, labels = [], []
    for class_name, class_idx in class_indices.items():
        class_dir = os.path.join(directory, class_name)  # Use class name correctly
        images = os.listdir(class_dir)
        selected_images = random.sample(images, min(max_images_per_class, len(images)))

        for img_name in selected_images:
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            data.append(img_array)
            labels.append(class_idx)

    return np.array(data), np.array(labels)


# Load class indices using ImageDataGenerator
class_indices = datagen.flow_from_directory(data_dir).class_indices

# Load random training data with 1000 images per category
max_images_per_class = 1000  # Limit each category to 1k images
train_data, train_labels = random_sample_images(data_dir, class_indices, max_images_per_class)

# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=len(class_indices))

# Load validation data normally
validation_data = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get the number of classes from the data generator
num_classes = validation_data.num_classes

# Define CNN model with a custom architecture
model = Sequential([
    tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),  # Explicitly set input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with the limited dataset
history = model.fit(
    train_data,
    train_labels,
    validation_data=validation_data,
    epochs=30,
    callbacks=[early_stopping]
)

# Save the trained model and label encoder
model.save('model/alzheimer_model.h5')

# Save the label encoder from the data generator class indices
joblib.dump(validation_data.class_indices, 'model/label_encoder.pkl')

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(validation_data)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
