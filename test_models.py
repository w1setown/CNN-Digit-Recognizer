#!/usr/bin/env python
"""Test script to verify model loading"""
import sys
import os

# Add src folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_ensemble import ModelEnsemble

print("Testing ModelEnsemble...")
ensemble = ModelEnsemble()
counts = ensemble.get_model_counts()
print(f"Models loaded - MNIST: {counts['mnist']}, EMNIST: {counts['emnist']}")

if counts['mnist'] > 0 or counts['emnist'] > 0:
    print("✓ Models loaded successfully!")
else:
    print("✗ No models found!")
