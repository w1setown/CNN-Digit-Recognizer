import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def load_and_prepare_mnist(use_emnist=False):
    """
    Load and preprocess the MNIST or EMNIST dataset
    """
    if use_emnist:
        try:
            # Load EMNIST using tensorflow-datasets
            dataset = tfds.load('emnist/digits', as_supervised=True)
            train_ds, test_ds = dataset['train'], dataset['test']
            
            # Convert to numpy arrays
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            
            for image, label in train_ds:
                x_train.append(image.numpy())
                y_train.append(label.numpy())
            
            for image, label in test_ds:
                x_test.append(image.numpy())
                y_test.append(label.numpy())
            
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            
        except Exception as e:
            print(f"Warning: EMNIST dataset loading failed ({str(e)}), falling back to MNIST")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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


def preprocess_image(image):
    """Preprocess a single image for prediction"""
    # Ensure image is grayscale and 28x28
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    
    # Normalize
    image = image.astype('float32') / 255
    
    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    return image