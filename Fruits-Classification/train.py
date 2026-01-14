import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# ayarlar veriyi cekme

DATA = "datasets/fruits-360_100x100/fruits-360/Training" #klasor yolu
IMG_SIZE = (100, 100)
BATCH_SIZE = 32

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

# Egitim verisi
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA,
    labels = 'inferred', #klasor isimlerini etiket yap
    label_mode = 'int',
    class_names = fruits_sets, #sadece bu on tanesini al
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    validation_split = 0.2, #yuzde 20 sini test icin ayir
    subset = "training",
    seed = 123
)

# Dogrulama (Validation) verisi
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA,
    class_names = fruits_sets,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    validation_split = 0.2,
    subset = "validation",
    seed = 123
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Toplam {num_classes} Farkli meyve turu bulundu")

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#CNN konvolusyonel sinir agi
model = tf.keras.models.Sequential([
    #resimdeki kenarlari ve sekilleri yakalar
    tf.keras.layers.Input(shape = (100, 100, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax') #10 meyve icin 10 cikis
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

print("Training Start...")
model.fit(train_ds, validation_data=val_ds, epochs=10)
print("Training Over")

model.save('fruits_model.h5')
print("model 'fruits_model.h5' olarak kaydedildi")