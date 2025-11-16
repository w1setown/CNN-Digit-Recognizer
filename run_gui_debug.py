#!/usr/bin/env python
"""
Test wrapper to run GUI and see console output.
"""
import sys
import os

# Change to the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

print(f"Working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

# Add src folder to path
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print(f"sys.path[0]: {sys.path[0]}")
print()

# Try to import and test
print("=" * 60)
print("Testing Model Loading")
print("=" * 60)

from model_ensemble import ModelEnsemble

ensemble = ModelEnsemble()
counts = ensemble.get_model_counts()
print(f"\nModel counts: {counts}")

if counts['mnist'] > 0 or counts['emnist'] > 0:
    print("✓ Models loaded successfully!")
    print("\nStarting GUI...")
    from gui import main
    main()
else:
    print("✗ No models found!")
    print("Please run: python create_models.py")
