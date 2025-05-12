import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
from model import build_cnn_model
from data_utils import load_and_prepare_mnist

class ModelEnsemble:
    def __init__(self):
        self.mnist_models = []
        self.emnist_models = []
        self.model_paths = []  # Initialize model_paths list
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load all existing models
        model_files = [f for f in os.listdir('models') if f.endswith('.keras')]
        
        for model_file in model_files:
            try:
                model_path = os.path.join('models', model_file)
                model = tf.keras.models.load_model(model_path)
                
                # Determine model type from filename
                if 'emnist' in model_file.lower():
                    self.emnist_models.append(model)
                else:
                    self.mnist_models.append(model)
                
                self.model_paths.append(model_path)  # Add path to model_paths
                print(f"Loaded model: {model_file}")
            except Exception as e:
                print(f"Error loading {model_file}: {e}")

    def add_model(self, model_path, dataset_type='mnist'):
        """Add a model to the ensemble"""
        if model_path not in self.model_paths:
            model = load_model(model_path)
            if dataset_type.lower() == 'mnist':
                self.mnist_models.append(model)
            else:
                self.emnist_models.append(model)
            self.model_paths.append(model_path)

    def create_new_model(self, images=None, labels=None, dataset_type='mnist', base_path='models/model'):
        """Create and train a new model
        
        Args:
            images: Optional training images. If None, uses MNIST/EMNIST dataset
            labels: Optional training labels. If None, uses MNIST/EMNIST dataset
            dataset_type (str): Type of dataset to use ('mnist' or 'emnist')
            base_path (str): Base path for saving the model
        """
        os.makedirs('models', exist_ok=True)
        
        # Create new model
        model = build_cnn_model()

        if images is None or labels is None:
            # Load appropriate dataset
            use_emnist = (dataset_type.lower() == 'emnist')
            (x_train, y_train), _ = load_and_prepare_mnist(use_emnist=use_emnist)
            
            # Find next available model number for this type
            prefix = 'emnist' if use_emnist else 'mnist'
            i = len(self.emnist_models if use_emnist else self.mnist_models)
            while os.path.exists(f"{base_path}_{prefix}_{i}.keras"):
                i += 1
            
            model_path = f"{base_path}_{prefix}_{i}.keras"
            
            # Train on dataset
            model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)
        else:
            # Train on provided data
            model.fit(images, labels, epochs=20, batch_size=32, verbose=1)
            
            # Save model with incrementing index
            model_count = len(self.mnist_models) + len(self.emnist_models)
            model_path = os.path.join('models', f'model_{model_count}.keras')

        # Save model
        model.save(model_path)
        
        # Add to ensemble
        self.add_model(model_path, dataset_type)
        print(f"Added new model to ensemble (total models: {len(self.mnist_models) + len(self.emnist_models)})")

    def predict(self, image):
        """Make prediction using all models in ensemble"""
        if not (self.mnist_models or self.emnist_models):
            raise ValueError("No models loaded in ensemble")
            
        predictions = []
        
        # Get predictions from MNIST models
        for model in self.mnist_models:
            pred = model.predict(image, verbose=0)
            predictions.append(pred[0])
            
        # Get predictions from EMNIST models 
        for model in self.emnist_models:
            pred = model.predict(image, verbose=0)
            predictions.append(pred[0])
        
        if predictions:
            # Average all predictions
            ensemble_prediction = np.mean(predictions, axis=0)
            return ensemble_prediction
        else:
            raise ValueError("No predictions generated")

    def get_model_counts(self):
        """Return the number of models of each type"""
        return {
            'mnist': len(self.mnist_models),
            'emnist': len(self.emnist_models)
        }