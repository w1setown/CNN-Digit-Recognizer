import numpy as np


def retrain_model(model, new_images, new_labels, model_path='cnn_model.keras'):
    """
    Fine-tune a model with new training examples
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