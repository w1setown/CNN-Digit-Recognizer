import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
from model import build_cnn_model
from data_utils import load_and_prepare_mnist
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

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
        
        # Setup callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        ]

        if images is None or labels is None:
            # Load appropriate dataset
            use_emnist = (dataset_type.lower() == 'emnist')
            (x_train, y_train), (x_test, y_test) = load_and_prepare_mnist(use_emnist=use_emnist)
            
            # Find next available model number for this type
            prefix = 'emnist' if use_emnist else 'mnist'
            i = len(self.emnist_models if use_emnist else self.mnist_models)
            while os.path.exists(f"{base_path}_{prefix}_{i}.keras"):
                i += 1
            
            model_path = f"{base_path}_{prefix}_{i}.keras"
            
            # Setup data augmentation for training robustness
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                fill_mode='nearest'
            )
            
            # Train with data augmentation
            datagen.fit(x_train)
            model.fit(
                datagen.flow(x_train, y_train, batch_size=128),
                epochs=15,
                validation_data=(x_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Create validation split from the provided data
            from sklearn.model_selection import train_test_split
            x_train, x_val, y_train, y_val = train_test_split(
                images, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Apply data augmentation for custom training data
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                fill_mode='nearest'
            )
            
            # Train on provided data with augmentation
            datagen.fit(x_train)
            model.fit(
                datagen.flow(x_train, y_train, batch_size=128),
                epochs=10,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
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