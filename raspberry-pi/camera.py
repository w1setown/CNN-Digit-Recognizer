import os
import numpy as np
import tensorflow as tf
import base64
import requests
import time


def load_model(model_path='cnn_model.keras'):
    """Load the TensorFlow model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)


def decode_preprocessed_image(preprocessed_b64, shape):
    """Decode base64 preprocessed image data back to numpy array"""
    image_bytes = base64.b64decode(preprocessed_b64)
    flat_array = np.frombuffer(image_bytes, dtype=np.float32)
    return flat_array.reshape(shape)


def predict_digit(model, preprocessed_image):
    """Make a prediction using the model"""
    prediction = model.predict(preprocessed_image, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return predicted_digit, confidence


def retrain_model(model, new_images, new_labels, model_path='cnn_model.keras'):
    """
    Fine-tune the model with new training examples
    """
    if not new_images:
        print("No new training data provided.")
        return model

    print("Retraining model with new examples...")

    # Convert lists to arrays if needed
    if isinstance(new_images, list):
        new_images = np.array(new_images)
    if isinstance(new_labels, list):
        new_labels = np.array(new_labels)

    # Fine-tune the model with new data
    model.fit(new_images, new_labels, epochs=3)

    # Save the updated model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model


def main():
    # Get the model
    model = load_model()

    # Storage for new training data
    new_training_images = []
    new_training_labels = []

    print("=== Continuous Camera Digit Recognition ===")
    print("Press Ctrl+C to exit")

    try:
        while True:
            input("\nPress Enter to capture an image (or Ctrl+C to exit)...")

            # Use camera to capture and process image
            try:
                print("Capturing image from camera...")
                response = requests.post('http://localhost:5000/api/capture')
                response.raise_for_status()

                data = response.json()
                if 'error' in data:
                    print(f"Error: {data['error']}")
                    continue

                # Decode the preprocessed image
                preprocessed_image = decode_preprocessed_image(
                    data['processed_image_b64'],
                    data['shape']
                )

                # Make prediction
                predicted_digit, confidence = predict_digit(model, preprocessed_image)
                print(f"Prediction: {predicted_digit} (Confidence: {confidence:.2f}%)")

                # Ask for correct label
                true_digit = input(
                    "What is the correct digit? (Press Enter if prediction was correct, or type the correct number): ")

                if true_digit and true_digit.isdigit():
                    # Add to training data
                    new_training_images.append(preprocessed_image[0])
                    new_training_labels.append(int(true_digit))

            except Exception as e:
                print(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        # Retrain if new data
        if new_training_images:
            print(f"\nCollected {len(new_training_images)} new training examples.")
            retrain_choice = input("Do you want to retrain the model with this data? (y/n): ")
            if retrain_choice.lower() == 'y':
                model = retrain_model(model, new_training_images, new_training_labels)


if __name__ == "__main__":
    main()