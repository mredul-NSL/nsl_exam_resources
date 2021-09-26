import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import cv2

image_directory = '/media/nsl4/hdd2/mredul/exam/mnist_synthetic/synthetic_images/resized_images/'


def preprocess(array):
    array = array.astype("float32") / 255.0
    #print(array.shape)
    array = np.reshape(array, (len(array), 28, 28, 1))
    #print(array.shape)
    return array


def clean(array):
    
    data = []

    for i in range(0,len(array)):
        img = cv2.imread(image_directory + str(array[i]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        img = img / 255.0
        #img = np.reshape(img, (28,28,1))
        #if i == 0:
            #print(array[i])
        data.append(img)

    data = np.reshape(data, (len(array), 28, 28, 1))
    #print(data.shape)

    return data


def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10
    #print("-------",array1.shape)
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    #print("-------",images1.shape)
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        #print("-------",image1.shape)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()



    
def train_model():
    
    (raw_train_data, train_labels), (raw_test_data, test_labels) = mnist.load_data()

    # Normalize and reshape the data
    noisy_train_data = preprocess(raw_train_data)
    noisy_test_data = preprocess(raw_test_data)

    clean_train_data = clean(train_labels)
    clean_test_data = clean(test_labels)
    NAME = "Mnist_to_synthetic{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    #model 
    input = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.summary()


    print(noisy_test_data.shape)
    print(noisy_train_data.shape)
    print(clean_test_data.shape)
    print(clean_train_data.shape)
    autoencoder.fit(
                x = noisy_train_data,
                y = clean_train_data,
                epochs=200,
                batch_size=128,
                shuffle=True,
                validation_split=0.3,
                callbacks=[tensorboard])

    
    predictions = autoencoder.predict(noisy_test_data)
    #print(predictions.shape)
    history = autoencoder.evaluate(noisy_test_data, clean_test_data, batch_size=128)

    autoencoder.save("auto-encoder-model")

    #test random samples
    display(noisy_test_data, predictions)

    # model summary
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    train_model()
    

main()
