# File Structure Reference

Complete directory and file organization of the CNN Digit Recognizer project.

## Project Root Directory

```
CNN-Digit-Recognizer/
├── .git/                              # Git repository (version control)
├── .github/                           # GitHub configuration
│   ├── workflows/                     # CI/CD workflows (if any)
│   └── ...
├── assets/                            # Images and resources
│   ├── flag_dk.png                    # Danish flag icon
│   ├── flag_uk.png                    # UK/English flag icon
│   └── logo.png                       # Application logo
├── src/                               # Main application source code
│   ├── __init__.py                    # Package initialization
│   ├── gui.py                         # Main GUI window & logic
│   ├── widgets.py                     # Custom UI widgets
│   ├── model.py                       # CNN model architecture
│   ├── model_ensemble.py              # Ensemble management
│   ├── data_utils.py                  # Data loading utilities
│   ├── digit_preprocessing.py         # Image preprocessing
│   ├── model_evaluation.py            # Model evaluation tools
│   └── __pycache__/                   # Python bytecode cache
├── models/                            # Trained model storage
│   ├── model_mnist_0.keras            # MNIST model 0
│   ├── model_mnist_1.keras            # MNIST model 1
│   ├── model_emnist_0.keras           # EMNIST model 0
│   └── ...
├── tests/                             # Unit tests
│   ├── __init__.py                    # Test package init
│   ├── test_model.py                  # Model tests
│   ├── test_data_utils.py             # Data utility tests
│   ├── test_digit_preprocessing.py    # Preprocessing tests
│   └── __pycache__/                   # Test bytecode cache
├── wiki/                              # Documentation (this folder)
│   ├── Home.md                        # Wiki home page
│   ├── Getting-Started.md             # Installation guide
│   ├── User-Guide.md                  # User manual
│   ├── Architecture.md                # System architecture
│   ├── Core-Modules.md                # Module documentation
│   ├── GUI-Components.md              # GUI documentation
│   ├── Model-System.md                # Model documentation
│   ├── Data-Processing.md             # Data processing guide
│   ├── File-Structure.md              # This file
│   ├── API-Documentation.md           # API reference
│   ├── Configuration.md               # Settings guide
│   ├── Development-Setup.md           # Dev environment
│   ├── Troubleshooting.md             # Problem solving
│   └── README.md                      # Wiki home
├── run_gui.py                         # Main application launcher
├── run_gui_debug.py                   # GUI with debug output
├── create_models.py                   # Model creation script
├── training.py                        # Training utilities
├── test_models.py                     # Model testing script
├── test_model_load.py                 # Model loading test
├── test_paths.py                      # Path testing
├── debug_paths.py                     # Path debugging
├── diagnose.py                        # System diagnostics
├── verify_setup.py                    # Setup verification
├── requirements.txt                   # Python dependencies
├── README.md                          # Project readme
├── MODEL_LOADING_FIX.md               # Model loading documentation
├── REORGANIZATION.md                  # Reorganization notes
└── .gitignore                         # Git ignore rules
```

## Directory Descriptions

### 📁 `src/` - Application Source Code

**Purpose:** Core application logic and features

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `__init__.py` | Package initialization | ~10 | Essential |
| `gui.py` | Main application window | ~369 | Core |
| `widgets.py` | Custom UI components | ~202 | Core |
| `model.py` | CNN architecture | ~70 | Core |
| `model_ensemble.py` | Model management | ~191 | Core |
| `data_utils.py` | Data loading/prep | ~80 | Core |
| `digit_preprocessing.py` | Image preprocessing | ~35 | Core |
| `model_evaluation.py` | Evaluation tools | Variable | Optional |

### 📁 `models/` - Trained Models

**Purpose:** Storage for Keras model files (.keras format)

**Contents:**
- Automatically created on first run
- Directory scanned at startup
- Each .keras file loaded as separate model
- File naming: `model_{dataset}_{index}.keras`

**Typical Size:**
- Single model: 50-100 MB
- Collection of 3 models: 150-300 MB

**Naming Convention:**
```
model_mnist_0.keras   → First MNIST model
model_mnist_1.keras   → Second MNIST model
model_emnist_0.keras  → First EMNIST model
model_emnist_1.keras  → Second EMNIST model
```

### 📁 `assets/` - Image Resources

**Purpose:** Images and icons for GUI

| File | Purpose | Size | Format |
|------|---------|------|--------|
| `flag_uk.png` | English flag icon | ~5 KB | PNG |
| `flag_dk.png` | Danish flag icon | ~5 KB | PNG |
| `logo.png` | Application logo | ~20 KB | PNG |

### 📁 `tests/` - Unit Tests

**Purpose:** Automated testing of core functionality

| File | Tests | Coverage |
|------|-------|----------|
| `test_model.py` | Model building, loading | CNN architecture |
| `test_data_utils.py` | Data loading, preprocessing | Data utilities |
| `test_digit_preprocessing.py` | Image preprocessing | Preprocessing pipeline |

**Running Tests:**
```bash
pytest tests/
# or
python -m pytest tests/ -v
```

### 📁 `wiki/` - Documentation

**Purpose:** Comprehensive project documentation

Complete wiki structure with 11+ documentation pages:
- User guides
- Developer guides
- API documentation
- Troubleshooting

See [Home.md](Home.md) for full wiki navigation.

---

## Utility Scripts

### `run_gui.py` - Main Application

```python
# Launches the GUI application
if __name__ == "__main__":
    from src.gui import DigitRecognitionApp
    app = DigitRecognitionApp()
    app.mainloop()
```

**Usage:**
```bash
python run_gui.py
```

### `run_gui_debug.py` - Debug Version

Enhanced version with debug output:

**Usage:**
```bash
python run_gui_debug.py
```

**Features:**
- Verbose logging
- Stack traces for errors
- Performance timing
- Prediction scores

### `create_models.py` - Model Creation

Creates initial trained models:

**Usage:**
```bash
python create_models.py
```

**What it does:**
1. Downloads MNIST dataset
2. Trains MNIST model
3. Downloads EMNIST dataset
4. Trains EMNIST model
5. Saves both to models/

**Time:** 10-30 minutes

### `training.py` - Custom Training

Additional training utilities and helpers.

**Purpose:** Advanced training scenarios

### `test_models.py` - Model Testing

Test script for model predictions:

**Usage:**
```bash
python test_models.py
```

### `test_model_load.py` - Load Testing

Tests model loading functionality:

**Usage:**
```bash
python test_model_load.py
```

### `verify_setup.py` - Setup Verification

Verifies installation is correct:

**Usage:**
```bash
python verify_setup.py
```

**Checks:**
- Python version
- Package installations
- Directory structure
- Model files
- File permissions

### `debug_paths.py` - Path Debugging

Debugs file path resolution:

**Usage:**
```bash
python debug_paths.py
```

### `diagnose.py` - System Diagnostics

Full system diagnostic report:

**Usage:**
```bash
python diagnose.py
```

**Reports:**
- Python version & executable
- Installed packages
- Directory structure
- Model availability
- GPU availability
- System memory

---

## Configuration Files

### `requirements.txt` - Dependencies

Lists all Python packages needed:

```
tensorflow>=2.13.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow-datasets>=4.9.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

### `.gitignore` - Git Ignore Rules

Prevents committing unnecessary files:

Typically excludes:
- `models/` - Large trained models
- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python
- `.venv/`, `venv/` - Virtual environments
- `.DS_Store` - macOS files
- `*.egg-info/` - Package info

### `.github/` - GitHub Configuration

Contains CI/CD workflows and GitHub-specific config:

**May include:**
- Automated tests on push
- Release workflows
- Issue templates
- PR templates

---

## Documentation Files

### `README.md` - Project README

Main project introduction:
- Feature overview
- Installation instructions
- Basic usage
- File structure
- Contribution guidelines

### `MODEL_LOADING_FIX.md` - Model Loading Notes

Documents model loading improvements and fixes:
- Loading issues encountered
- Solutions implemented
- Path resolution strategies

### `REORGANIZATION.md` - Reorganization Notes

Documents project structure changes:
- Original structure
- Changes made
- Rationale for changes
- Migration notes

---

## Key Paths & Locations

### Application Startup Path

```
run_gui.py
    ↓
src/gui.py::DigitRecognitionApp.__init__()
    ↓
Loads models from: ../models/
    ↓
Creates ModelEnsemble
    ↓
GUI window appears
```

### Model Discovery Path

```
models/ directory
    ↓
ModelEnsemble.__init__()
    ↓
Scan for *.keras files
    ↓
Load each model
    ↓
Categorize (MNIST/EMNIST)
    ↓
Add to ensemble
```

### Image Processing Path

```
Canvas image (280×280)
    ↓
digit_preprocessing.preprocess_digit_image()
    ↓
Binary → Contours → Extract → Pad → Normalize
    ↓
Model input (28×28×1)
    ↓
ModelEnsemble.predict()
    ↓
Display results
```

---

## File Size Summary

| Component | Typical Size |
|-----------|--------------|
| Source code (src/) | ~1 MB |
| Single model | 50-100 MB |
| 3-model ensemble | 150-300 MB |
| Assets | ~50 KB |
| Tests | ~100 KB |
| Documentation | ~500 KB |
| **Total** | **~600 MB minimum** |

---

## Important Files for Development

### If you want to modify...

| Feature | Edit File(s) |
|---------|--------------|
| **GUI appearance** | `src/gui.py`, `src/widgets.py` |
| **CNN architecture** | `src/model.py` |
| **Image preprocessing** | `src/digit_preprocessing.py` |
| **Model management** | `src/model_ensemble.py` |
| **Data loading** | `src/data_utils.py` |
| **Training process** | `src/model_ensemble.py` |
| **UI text/labels** | `src/gui.py` |
| **Languages** | `src/gui.py` |
| **Colors/styling** | `src/gui.py`, `src/widgets.py` |

---

See also: [Architecture Overview](Architecture.md), [Core Modules](Core-Modules.md), [Development Setup](Development-Setup.md)
