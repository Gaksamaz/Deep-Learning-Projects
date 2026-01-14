import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('fruits_model.h5')

# 2. Class names list 
class_names = [
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

# Image path
img_path = 'fruits/cabbage.jpeg'

try:
    # Load and resize the image
    img = load_img(img_path, target_size=(100, 100))
    
    # Convert image to array and normalize pixel values
    img_array = img_to_array(img) / 255.0
    
    # Expand dimensions to match model input shape (batch size = 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    predictions = model.predict(img_array)
    
    # Print raw model outputs (for debugging purposes)
    print(f"Raw Predictions: {predictions}") 
    
    # Get the index of the class with the highest confidence
    score_index = np.argmax(predictions[0])
    
    # Get predicted class name and confidence score
    predicted_class = class_names[score_index]
    confidence_score = 100 * predictions[0][score_index]

    print("-" * 30)
    print(f"Result: {predicted_class}")
    print(f"Accuracy: %{confidence_score:.2f}")
    print("-" * 30)

except Exception as e:
    # Handle and print any errors
    print(f"ERROR!: {e}")
