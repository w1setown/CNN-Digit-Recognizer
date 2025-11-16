#!/usr/bin/env python
"""
Launcher script for the CNN Digit Recognizer GUI.

This script runs the main GUI application from the src module.
Run this from the project root to start the application.
"""
import sys
import os

# Ensure we're in the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Add src folder to path
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from gui import main

if __name__ == "__main__":
    main()
