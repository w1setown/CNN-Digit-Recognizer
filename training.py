import numpy as np


def retrain_model(model, new_images, new_labels, model_path='cnn_model.keras', callbacks=None):
    """
    Fine-tune a model with new training examples
    
    Args:
        model: The keras model to retrain
        new_images: Training images
        new_labels: Training labels
        model_path: Path to save the model
        callbacks: List of Keras callbacks to apply during training (default: None)
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

    # Create default callbacks list if none provided
    if callbacks is None:
        callbacks = []

    # Fine-tune the model with new data
    history = model.fit(new_images, new_labels, 
                       epochs=20,  # Increased epochs 
                       verbose=1,
                       callbacks=callbacks)

    # Save the updated model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model