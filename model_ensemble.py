import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
from model import build_cnn_model

class ModelEnsemble:
    def __init__(self, base_model_path='cnn_model.keras'):
        self.models = []
        self.model_paths = []
        
        # Load base model if exists
        if os.path.exists(base_model_path):
            self.add_model(base_model_path)
    
    def add_model(self, model_path):
        """Add a model to the ensemble"""
        if model_path not in self.model_paths:
            model = load_model(model_path)
            self.models.append(model)
            self.model_paths.append(model_path)
    
    def create_new_model(self, images, labels, base_path='models/model'):
        """Create and train a new model"""
        os.makedirs('models', exist_ok=True)
        
        # Find next available model number
        i = len(self.models)
        while os.path.exists(f"{base_path}_{i}.keras"):
            i += 1
        
        model_path = f"{base_path}_{i}.keras"
        
        # Create and train new model
        model = build_cnn_model()
        model.fit(images, labels, epochs=20, batch_size=32)
        
        # Save model
        model.save(model_path)
        
        # Add to ensemble
        self.add_model(model_path)
    
    def predict(self, image):
        """Make prediction using all models in ensemble"""
        if not self.models:
            raise ValueError("No models in ensemble")
            
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(image, verbose=0)
            predictions.append(pred[0])
            
        # Average the predictions
        ensemble_prediction = np.mean(predictions, axis=0)
        return ensemble_prediction