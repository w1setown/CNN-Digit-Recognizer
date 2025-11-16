#!/usr/bin/env python
"""
Comprehensive diagnostic script for CNN Digit Recognizer
"""
import os
import sys

print("=" * 70)
print("CNN DIGIT RECOGNIZER - DIAGNOSTIC SCRIPT")
print("=" * 70)

# 1. Check Python version and environment
print(f"\n[1] Python Environment:")
print(f"  Python version: {sys.version}")
print(f"  Python executable: {sys.executable}")
print(f"  Current working directory: {os.getcwd()}")

# 2. Check directory structure
print(f"\n[2] Directory Structure:")
project_root = os.path.dirname(os.path.abspath(__file__))
print(f"  Project root: {project_root}")

directories = ['src', 'assets', 'models', 'tests']
for d in directories:
    d_path = os.path.join(project_root, d)
    exists = os.path.exists(d_path)
    print(f"  {d}/: {'✓' if exists else '✗'} {d_path}")

# 3. Check model files
print(f"\n[3] Model Files:")
models_dir = os.path.join(project_root, 'models')
if os.path.exists(models_dir):
    files = os.listdir(models_dir)
    keras_files = [f for f in files if f.endswith('.keras')]
    print(f"  Total files: {len(files)}")
    print(f"  Keras files: {len(keras_files)}")
    for f in keras_files:
        f_path = os.path.join(models_dir, f)
        f_size = os.path.getsize(f_path)
        print(f"    - {f} ({f_size} bytes)")
else:
    print(f"  ✗ Models directory not found!")

# 4. Check imports
print(f"\n[4] Module Imports:")
sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    import tensorflow as tf
    print(f"  ✓ tensorflow: {tf.__version__}")
except ImportError as e:
    print(f"  ✗ tensorflow: {e}")

try:
    import numpy as np
    print(f"  ✓ numpy: {np.__version__}")
except ImportError as e:
    print(f"  ✗ numpy: {e}")

try:
    import cv2
    print(f"  ✓ opencv-python: {cv2.__version__}")
except ImportError as e:
    print(f"  ✗ opencv-python: {e}")

try:
    from PIL import Image
    print(f"  ✓ pillow")
except ImportError as e:
    print(f"  ✗ pillow: {e}")

# 5. Test model ensemble initialization
print(f"\n[5] ModelEnsemble Test:")
try:
    from model_ensemble import ModelEnsemble
    print(f"  ✓ ModelEnsemble imported")
    
    ensemble = ModelEnsemble()
    counts = ensemble.get_model_counts()
    print(f"  ✓ ModelEnsemble created")
    print(f"    - MNIST models: {counts['mnist']}")
    print(f"    - EMNIST models: {counts['emnist']}")
    
    if counts['mnist'] > 0 or counts['emnist'] > 0:
        print(f"  ✓ Models loaded successfully!")
    else:
        print(f"  ✗ No models loaded!")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
