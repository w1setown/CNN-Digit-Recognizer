#!/usr/bin/env python
"""
Verification script - mimics exact environment of run_gui.py
"""
import sys
import os

# Replicate run_gui.py setup
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print("=" * 70)
print("VERIFICATION: Exact run_gui.py Environment")
print("=" * 70)
print(f"\nProject root: {project_root}")
print(f"Working directory: {os.getcwd()}")
print(f"sys.path[0]: {sys.path[0]}")

print("\n[TEST 1] Importing modules...")
try:
    from model_ensemble import ModelEnsemble
    print("✓ model_ensemble imported")
except Exception as e:
    print(f"✗ Failed to import model_ensemble: {e}")
    sys.exit(1)

print("\n[TEST 2] Creating ModelEnsemble...")
try:
    ensemble = ModelEnsemble()
    print("✓ ModelEnsemble created")
except Exception as e:
    print(f"✗ Failed to create ModelEnsemble: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[TEST 3] Checking model counts...")
counts = ensemble.get_model_counts()
print(f"  MNIST models: {counts['mnist']}")
print(f"  EMNIST models: {counts['emnist']}")

if counts['mnist'] > 0 or counts['emnist'] > 0:
    print("✓ Models loaded successfully!")
    
    print("\n[TEST 4] Testing prediction...")
    import numpy as np
    try:
        # Create a dummy image (28x28x1)
        dummy_image = np.random.rand(1, 28, 28, 1).astype('float32')
        prediction = ensemble.predict(dummy_image)
        print(f"✓ Prediction successful!")
        print(f"  Output shape: {prediction.shape}")
        print(f"  Sample values: {prediction[:3]}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("✗ NO MODELS LOADED!")
    print("\nYou need to create models first:")
    print("  python create_models.py")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - Application should work!")
print("=" * 70)
print("\nYou can now run: python run_gui.py")
