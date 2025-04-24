import cv2
import numpy as np
import tensorflow as tf


def load_and_prepare_mnist(use_emnist=True):
    """
    Load and preprocess the MNIST or EMNIST dataset
    """
    if use_emnist:
        # Load EMNIST digits dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.emnist.load_data(subset='digits')
        
        # EMNIST images need to be rotated and flipped
        x_train = np.rot90(x_train, k=3, axes=(1,2))
        x_train = np.flip(x_train, axis=1)
        x_test = np.rot90(x_test, k=3, axes=(1,2))
        x_test = np.flip(x_test, axis=1)
    else:
        # Original MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape for CNN input (add channel dimension)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)


# def load_and_prepare_mnist():
#     """
#     Load and preprocess the MNIST dataset
#     """
#     mnist = tf.keras.datasets.mnist
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#     # Reshape for CNN input (add channel dimension)
#     x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#     x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#
#     # Normalize pixel values
#     x_train = x_train.astype('float32') / 255
#     x_test = x_test.astype('float32') / 255
#
#     return (x_train, y_train), (x_test, y_test)


def preprocess_image(img_path):
    """
    Preprocess an image for prediction
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28 if needed
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Invert if needed (assuming white digit on black background)
    if np.mean(img) > 127:
        img = 255 - img

    # Normalize and reshape for CNN
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img