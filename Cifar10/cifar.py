import numpy as np
import keras
from keras import layers,models

dataset = keras.utils.image_dataset_from_directory('cifar10/',labels=None,image_size=(32,32),batch_size=32,shuffle=True)


dataset = dataset.map(lambda x: (x / 255.0, x / 255.0))
input_img = keras.Input(shape=(32, 32, 3))
# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(dataset, epochs=30)

autoencoder.save('cifar10.keras')