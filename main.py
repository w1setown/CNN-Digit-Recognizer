import os
import numpy as np

# Import from other modules
from model import load_or_train_model
from data_utils import preprocess_image
from visualization import display_prediction_results
from training import retrain_model


def main():
    # Get the model
    model = load_or_train_model()

    image_number = 1
    new_training_images = []
    new_training_labels = []

    while os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            print(f"\nAnalyzing digit{image_number}.png...")

            # Preprocess the image
            img_path = f"digits/digit{image_number}.png"
            processed_img = preprocess_image(img_path)

            # Make prediction
            prediction = model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            print(f"This digit is probably a {predicted_digit} (confidence: {confidence:.2f}%)")

            # Display the image with prediction
            display_prediction_results(processed_img, predicted_digit, image_number, confidence)

            # Ask for correct label
            true_digit = input(
                "What is the correct digit? (Press Enter if prediction was correct, or type the correct number): ")

            if true_digit and true_digit.isdigit():
                # Add to training data
                new_training_images.append(processed_img[0])
                new_training_labels.append(int(true_digit))

        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            image_number += 1

    # Retrain if we collected new training data
    if new_training_images:
        model = retrain_model(model, new_training_images, new_training_labels)

    print("All digits have been processed.")


if __name__ == "__main__":
    main()