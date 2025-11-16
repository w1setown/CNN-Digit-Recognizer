#!/usr/bin/env python
"""Simple test to verify paths in model ensemble"""
import os
import sys

# First, let's see what __file__ would be when run from src
test_file_path = r'c:\Users\68691\OneDrive - Syddansk Erhvervsskole\Desktop\CNN-Digit-Recognizer\src\model_ensemble.py'

src_dir = os.path.dirname(os.path.abspath(test_file_path))
models_dir = os.path.join(os.path.dirname(src_dir), 'models')

print(f"test_file_path: {test_file_path}")
print(f"src_dir: {src_dir}")
print(f"os.path.dirname(src_dir): {os.path.dirname(src_dir)}")
print(f"models_dir: {models_dir}")
print(f"models_dir exists: {os.path.exists(models_dir)}")

if os.path.exists(models_dir):
    files = os.listdir(models_dir)
    print(f"Files in models_dir: {files}")
    keras_files = [f for f in files if f.endswith('.keras')]
    print(f"Keras files: {keras_files}")
