import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #type: ignore


def build_cnn_model():
    """
    Build and return a CNN model for digit recognition
    """
    model = Sequential([
        # First Convolutional Layer
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Layer
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten the results for the dense layers
        Flatten(),

        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.2),  # Dropout to prevent overfitting
        Dense(10, activation='softmax')  # 10 output classes (digits 0-9)
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_or_train_model(model_path='cnn_model.keras'):
    """
    Load an existing model or train a new one if none exists
    """
    import os
    from data_utils import load_and_prepare_mnist

    if os.path.exists(model_path):
        print("Loading existing CNN model...")
        return tf.keras.models.load_model(model_path)
    else:
        print("Training new CNN model...")
        model = build_cnn_model()

        # Load and prepare MNIST data
        (x_train, y_train), (x_test, y_test) = load_and_prepare_mnist()

        # Train model
        model.fit(
            x_train, y_train,
            epochs=5,
            validation_data=(x_test, y_test),
            batch_size=128
        )

        # Evaluate model
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {accuracy:.4f}")

        # Save model
        model.save(model_path)
        return model