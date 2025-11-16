#!/usr/bin/env python
"""Debug script to check model paths"""
import sys
import os

# Add src folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("DEBUG: Path Resolution")
print("=" * 60)

# Simulate what happens in model_ensemble.py
src_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
print(f"src_dir would be: {src_dir}")

# This is what the code will calculate
models_dir_parent = os.path.dirname(src_dir)
models_dir = os.path.join(models_dir_parent, 'models')
print(f"models_dir would be: {models_dir}")

# Check if it exists and what's in it
if os.path.exists(models_dir):
    print(f"✓ Models directory exists")
    files = os.listdir(models_dir)
    print(f"  Files in directory: {files}")
else:
    print(f"✗ Models directory does not exist")

print("=" * 60)
