# Getting Started Guide

Quick start guide for users who want to set up and run the CNN Digit Recognizer application.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.7 or higher** (3.11 recommended)
- **pip** package manager
- **4 GB RAM minimum** (8 GB recommended)
- **500 MB disk space** for models and data
- **Windows, macOS, or Linux**

### Check Your Python Installation

**Windows:**
```powershell
python --version
```

**macOS/Linux:**
```bash
python3 --version
```

Should output something like: `Python 3.11.x`

## Installation Steps

### 1. Clone the Repository

**Windows (PowerShell):**
```powershell
git clone https://github.com/w1setown/CNN-Digit-Recognizer
cd CNN-Digit-Recognizer
```

**macOS/Linux:**
```bash
git clone https://github.com/w1setown/CNN-Digit-Recognizer
cd CNN-Digit-Recognizer
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear before your command prompt.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages installed:**
- TensorFlow 2.13+ - Deep learning framework
- OpenCV (cv2) - Image processing
- Pillow - Image manipulation
- Tkinter - GUI framework
- Matplotlib - Visualization
- NumPy - Numerical computing
- Scikit-learn - Machine learning utilities
- TensorFlow Datasets - Dataset loading

### 4. Create Models Directory

```powershell
mkdir models
```

### 5. (Optional) Create Initial Models

If you want pre-trained models:

```bash
python create_models.py
```

⏱️ **Note:** This may take 10-30 minutes depending on your system.

If you skip this, models will be created automatically on first use.

## Running the Application

### Start the GUI

```bash
python run_gui.py
```

A window should appear with:
- White drawing canvas (left side)
- Prediction display with chart (right side)
- Training controls (bottom)
- Action buttons

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Clear canvas |
| `Enter` or `Ctrl+X` | Make prediction |
| `Ctrl+V` | Add drawing to training data |
| `Ctrl+N` | Train new model |
| Click flag icon | Switch language (English/Danish) |

## First Run Checklist

- [ ] Application window appears
- [ ] Canvas is white and responsive
- [ ] Can draw with mouse
- [ ] Drawing appears as black strokes
- [ ] Can clear canvas (Ctrl+Z)
- [ ] Can make predictions (Enter)
- [ ] Prediction and confidence displayed
- [ ] Chart shows probabilities for digits 0-9

## Troubleshooting First Run

### "Module not found" Error

**Problem:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
# Ensure virtual environment is activated
# (should see (venv) in command prompt)

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### "No models found" Warning

**Problem:** Application starts but gives warning about missing models

**Solution:**
```bash
# Option 1: Create models (takes time)
python create_models.py

# Option 2: Download pre-trained models
# (See https://github.com/w1setown/CNN-Digit-Recognizer/releases)

# Application will still work with fewer models,
# but accuracy will be lower
```

### GUI Doesn't Appear

**Problem:** Application starts but no window visible

**Possible causes:**
- Tkinter not installed
- Display server issue (Linux)
- Window rendered off-screen

**Solutions:**

**Windows:**
```powershell
# Reinstall Tkinter
python -m pip install --upgrade tk
```

**Linux:**
```bash
# Install Tkinter system package
sudo apt-get install python3-tk

# Then reinstall Python packages
pip install --upgrade -r requirements.txt
```

**macOS:**
```bash
# Tkinter usually included with Python
# If missing, reinstall Python from python.org or use Homebrew
```

### Slow Predictions

**Problem:** Predictions take 2+ seconds

**Possible causes:**
- First prediction slower (models loading)
- System CPU limited
- Too many background programs

**Solutions:**
- Subsequent predictions should be faster
- Close other applications
- Ensure adequate RAM available
- Consider GPU acceleration (advanced)

### Poor Prediction Accuracy

**Problem:** Predictions often wrong

**Possible causes:**
- Handwriting style different from training data
- Digit not clearly drawn
- Model not trained
- Preprocessing issue

**Solutions:**
1. Draw digit more clearly
2. Ensure digit fills 60-80% of canvas
3. Draw roughly centered
4. Use drawing tablet if available for better control
5. Check if models exist: `dir models/`
6. Try contributing more training data

## Next Steps

Now that you have the application running:

1. **Learn the UI** - See [User Guide](User-Guide.md)
2. **Understand the code** - See [Architecture Overview](Architecture.md)
3. **Contribute training data** - Draw digits to improve accuracy
4. **Explore settings** - See [Configuration](Configuration.md)
5. **Develop further** - See [Development Setup](Development-Setup.md)

## System Requirements

### Minimum
- Python 3.7
- 4 GB RAM
- 500 MB disk space
- Intel/AMD processor

### Recommended
- Python 3.11
- 8 GB RAM
- 2 GB disk space (including models)
- Modern multi-core processor

### GPU Support (Optional)

For faster predictions with NVIDIA GPU:

```bash
# Uninstall CPU-only TensorFlow
pip uninstall tensorflow

# Install GPU version
pip install tensorflow[and-cuda]
```

See [TensorFlow GPU setup](https://www.tensorflow.org/install/gpu) for details.

## Getting Help

If you encounter issues:

1. **Check the log output** - Error messages usually indicate the problem
2. **Review [Troubleshooting](Troubleshooting.md)** - Common issues and solutions
3. **Check project issues** - [GitHub Issues](https://github.com/w1setown/CNN-Digit-Recognizer/issues)
4. **Consult documentation** - This wiki has detailed explanations

## What's Included

After installation, you'll have:

```
CNN-Digit-Recognizer/
├── src/                    # Application source code
│   ├── gui.py             # Main application
│   ├── model_ensemble.py  # Model management
│   ├── widgets.py         # UI components
│   ├── model.py           # CNN architecture
│   ├── data_utils.py      # Data utilities
│   └── digit_preprocessing.py  # Image preprocessing
├── models/                # Your trained models
├── assets/                # Images and resources
├── run_gui.py            # Application launcher
├── create_models.py      # Model creation script
├── requirements.txt      # Package dependencies
└── wiki/                 # This documentation
```

## Privacy & Data

- **Canvas drawings** are stored locally only during your session
- **Training data** is saved to local `models/` directory when you add it
- **No data** is sent to external servers
- **Models** are standard Keras files - you have full control

## Next: [User Guide](User-Guide.md)

Ready to use the application? Check out the [User Guide](User-Guide.md) for detailed feature documentation.

---

**Last Updated:** November 2025  
**Python Version:** 3.7+  
**TensorFlow Version:** 2.13+
