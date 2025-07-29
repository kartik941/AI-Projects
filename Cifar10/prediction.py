import numpy as np
from keras.preprocessing import image
import keras
import matplotlib.pyplot as plt

img_path = "../dog_cat_clasifier/dataset/test_set/cats/cat.4095.jpg"
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize if model was trained with ImageDataGenerator(rescale=1./255)


# Load model
autoencoder = keras.models.load_model("cifar10.keras")


reconstructed_img = autoencoder.predict(img_array)

# Visualize
plt.subplot(1, 2, 1)
plt.imshow(img_array[0])
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img[0])
plt.title("Reconstructed")
plt.axis('off')

plt.show()


