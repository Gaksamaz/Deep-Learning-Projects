import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Dataset configuration

DATA = "datasets/fruits-360_100x100/fruits-360/Training"  # Dataset directory path
IMG_SIZE = (100, 100)
BATCH_SIZE = 32

# Selected fruit classes
fruits_sets = [
    'Avocado 1', 
    'Banana 1',
    'Cabbage red 1', 
    'Cherry 1', 
    'Kiwi 1', 
    'Mandarine 1', 
    'Strawberry 1',
    'Tomato 3',  
    'Walnut 1',
    'Watermelon 1'
]

# Training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA,
    labels='inferred',              # Use folder names as labels
    label_mode='int',
    class_names=fruits_sets,        # Use only selected classes
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,           # Reserve 20% of data for validation
    subset="training",
    seed=123
)

# Validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA,
    class_names=fruits_sets,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Extract class names and number of classes
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Total number of fruit classes found: {num_classes}")

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Convolutional Neural Network (CNN) model
model = tf.keras.models.Sequential([
    # Input layer
    tf.keras.layers.Input(shape=(100, 100, 3)),
    
    # Convolution and pooling layers to extract edges and shapes
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Fully connected layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # One output per fruit class
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training started...")
model.fit(train_ds, validation_data=val_ds, epochs=10)
print("Training completed.")

# Save the trained model
model.save('fruits_model.h5')
print("Model saved as 'fruits_model.h5'")
