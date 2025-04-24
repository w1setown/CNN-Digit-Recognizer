import numpy as np


def retrain_model(model, new_images, new_labels, model_path='cnn_model.keras'):
    """
    Fine-tune a model with new training examples
    """
    if len(new_images) == 0:
        print("No new training data provided.")
        return model

    print(f"Retraining model with {len(new_images)} new examples...")

    # Convert lists to arrays if needed
    if isinstance(new_images, list):
        new_images = np.array(new_images)
    if isinstance(new_labels, list):
        new_labels = np.array(new_labels)

    # Fine-tune the model with new data
    # Use the same loss that was used during model compilation
    history = model.fit(new_images, new_labels, epochs=3, verbose=1)

    # Save the updated model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model