import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Modeli yükle
model = tf.keras.models.load_model('fruits_model.h5')

# 2. Sınıf isimlerini DÜZELTİLMİŞ liste (Virgüllere dikkat!)
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

# Resim yolu
img_path = 'fruits/cabbage.jpeg'

try:
    img = load_img(img_path, target_size=(100, 100))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    
    # Modelin ham çıktılarını görmek için (Hata ayıklama amaçlı)
    print(f"Ham Tahminler: {predictions}") 
    
    score_index = np.argmax(predictions[0])
    tahmin_edilen = class_names[score_index]
    guven_orani = 100 * predictions[0][score_index]

    print("-" * 30)
    print(f"Result: {tahmin_edilen}")
    print(f"Accuracy: %{guven_orani:.2f}")
    print("-" * 30)

except Exception as e:
    print(f"ERROR!: {e}")