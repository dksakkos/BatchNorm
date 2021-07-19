from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., np.newaxis]
    y_train = np_utils.to_categorical(y_train, 10)
    x_test = x_test[..., np.newaxis]
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def get_train_gen():
    return ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=20,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        validation_split=0.1)


def get_test_gen():
    return ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
