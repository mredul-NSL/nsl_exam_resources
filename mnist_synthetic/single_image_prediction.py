import tensorflow as tf 
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_directory = '/media/nsl4/hdd2/mredul/exam/mnist_synthetic/test_images/'

img = cv2.imread(image_directory + "sample_image.png", cv2.IMREAD_GRAYSCALE)
img = np.array(img)

img = cv2.resize(img, (28, 28))
img = img / 255.

img = np.reshape(img, (1,28,28,1))
#plt.imshow(np.reshape(img,(28,28)))

model = tf.keras.models.load_model("auto-encoder-model")
predictions = model.predict(img)
print(predictions.shape)
plt.imshow(np.reshape(predictions,(28,28)))
plt.show()

