#!/usr/bin/env python
"""Test if model file can be loaded"""
import os
import tensorflow as tf

model_path = r'c:\Users\68691\OneDrive - Syddansk Erhvervsskole\Desktop\CNN-Digit-Recognizer\models\model_mnist_0.keras'

print(f"Model path: {model_path}")
print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'}")

if os.path.exists(model_path):
    try:
        print("Attempting to load model...")
        model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully!")
        print(f"  Model summary:")
        model.summary()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ Model file not found!")
