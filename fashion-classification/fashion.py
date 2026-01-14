import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Veriyi internetten çekiyoruz (Tek satır!)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Kıyafet isimlerini tanımlayalım (Etiketler 0-9 arası sayıdır)
class_names = ['T-shirt', 'Pantolon', 'Kazak', 'Elbise', 'Ceket', 
               'Sandalet', 'Gömlek', 'Sneaker', 'Çanta', 'Bot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    # 28x28'lik kare resmi tek bir uzun şerit (784 piksel) yapar
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # Gizli Katman: 128 nöronlu, desenleri yakalayan bölüm (dusunen kisim)
    keras.layers.Dense(128, activation='relu'), #relu aktivasyonu onemli bilgileri gecirir (0in altindakileri engeller)
    
    # Çıkış Katmanı: 10 nöron (Her biri bir kıyafet türünü temsil eder)
    keras.layers.Dense(10, activation='softmax')
])

#compile -> kurallari belirle
model.compile(optimizer='adam', #(matematiksel formul) hata payini azaltmak icin noronlar arasindaki baglari guclendirir
              loss='sparse_categorical_crossentropy', #AI fonksiyonun ne kadar buyuk hata yaptigini hesaplar
              metrics=['accuracy'])

#fit -> ogren
model.fit(train_images, train_labels, epochs=10) #AI in bastan sona 10 kez incelemesini saglar
tahminler = model.predict(test_images) #predict -> tahmin et
index = 0 # İlk test resmini seçelim

print(f"Gerçek Etiket: {class_names[test_labels[index]]}")
print(f"AI Tahmini: {class_names[np.argmax(tahminler[index])]}")

# Resmi görelim
plt.imshow(test_images[index], cmap=plt.cm.binary)
plt.show()

plt.imshow(train_images[index], cmap=plt.cm.binary)
plt.show()

#train images ai in ders calistigi sorular 
#test images ai in sinav oldugu sorular
