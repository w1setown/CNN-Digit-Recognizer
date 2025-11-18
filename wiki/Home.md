# CNN Digit Recognizer Wiki

Welcome to the **CNN Digit Recognizer** project wiki! This is your comprehensive guide to understanding, developing, and maintaining the handwritten digit recognition GUI application.

## 📚 Quick Navigation

### For Users
- **[Getting Started](Getting-Started.md)** - Installation, setup, and first run
- **[User Guide](User-Guide.md)** - How to use the application features
- **[Keyboard Shortcuts](User-Guide.md#keyboard-shortcuts)** - Quick reference

### For Developers
- **[Architecture Overview](Architecture.md)** - System design and component relationships
- **[Core Modules](Core-Modules.md)** - Detailed documentation of each Python module
- **[GUI Components](GUI-Components.md)** - UI widgets and layout
- **[Model System](Model-System.md)** - CNN models, ensemble architecture, and training
- **[Data Processing](Data-Processing.md)** - Image preprocessing and data utilities
- **[Development Setup](Development-Setup.md)** - Setting up a development environment

### Reference
- **[File Structure](File-Structure.md)** - Complete project directory layout
- **[API Documentation](API-Documentation.md)** - Function and class reference
- **[Configuration](Configuration.md)** - Settings and customization
- **[Troubleshooting](Troubleshooting.md)** - Common issues and solutions

## 🎯 Project Overview

**CNN Digit Recognizer** is an interactive desktop application that:
- Allows users to draw handwritten digits (0-9) on a canvas
- Uses ensemble CNN models for real-time digit recognition
- Displays prediction confidence for all digit classes
- Enables users to contribute training data to improve accuracy
- Supports both English and Danish interfaces

### Key Features
✅ **Real-time Preprocessing** - Live visual feedback on MNIST-formatted input  
✅ **Ensemble Predictions** - Multiple models for improved accuracy  
✅ **Interactive Training** - Add your drawings to the training dataset  
✅ **Multilingual UI** - English and Danish interface support  
✅ **Model Management** - Automatic model loading and creation  

## 📦 Tech Stack

- **Python 3.11** - Core language
- **TensorFlow/Keras** - Deep learning framework
- **Tkinter** - GUI framework
- **OpenCV** - Image processing
- **Pillow** - Image manipulation
- **Scikit-learn** - Machine learning utilities
- **Matplotlib** - Visualization
- **NumPy** - Numerical computing

## 🏗️ Architecture at a Glance

```
User Interface (Tkinter GUI)
    ↓
Model Ensemble System
    ├── MNIST Models
    └── EMNIST Models
    ↓
Preprocessing Pipeline
    ├── Image Capture (Canvas)
    ├── Binary Conversion
    ├── Digit Extraction
    ├── Normalization
    └── MNIST Format (28x28x1)
    ↓
Prediction & Display
    ├── Confidence Scores
    └── Training Feedback
```

## 🚀 Quick Start

### For Users
```bash
pip install -r requirements.txt
python run_gui.py
```

### For Developers
```bash
git clone https://github.com/w1setown/CNN-Digit-Recognizer
cd CNN-Digit-Recognizer
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python create_models.py  # Optional: Create initial models
python run_gui.py
```

## 📋 Important Files & Directories

| Path | Purpose |
|------|---------|
| `src/gui.py` | Main application window and logic |
| `src/model_ensemble.py` | Ensemble model management |
| `src/widgets.py` | UI components (canvas, charts, panels) |
| `src/model.py` | CNN model architecture |
| `src/data_utils.py` | Data loading and preprocessing |
| `src/digit_preprocessing.py` | Advanced digit preprocessing |
| `models/` | Trained model storage (.keras files) |
| `assets/` | Images and resources |

## 🔗 Related Documentation

- **[MODEL_LOADING_FIX.md](../MODEL_LOADING_FIX.md)** - Model loading improvements
- **[REORGANIZATION.md](../REORGANIZATION.md)** - Project structure changes
- **[README.md](../README.md)** - Main project README

## 📞 Support & Contributing

- **Issues & Bugs** - See [Troubleshooting](Troubleshooting.md)
- **Contributing** - Check individual module documentation for code patterns
- **Questions** - Refer to relevant wiki pages for your use case

---

**Last Updated:** November 2025  
**Project Repository:** [github.com/w1setown/CNN-Digit-Recognizer](https://github.com/w1setown/CNN-Digit-Recognizer)
