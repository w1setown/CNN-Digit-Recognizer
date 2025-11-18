# Core Modules Reference

Detailed documentation of all core Python modules in the CNN Digit Recognizer project.

## 🎨 gui.py - Main Application

**Location:** `src/gui.py`  
**Purpose:** Main application window, event handling, and UI logic  
**Lines:** ~369

### Key Classes

#### `ToolTip`
Simple tooltip widget that displays help text on hover.

**Methods:**
- `show_tip(event)` - Display tooltip at mouse position
- `hide_tip(event)` - Hide tooltip
- `set_text(text)` - Update tooltip text

#### `DigitRecognitionApp(tk.Tk)`
Main application window class inheriting from Tkinter's Tk root window.

**Attributes:**
- `canvas` (DrawingCanvas) - Digit drawing area
- `model_ensemble` (ModelEnsemble) - Loaded CNN models
- `training_data` (list) - User-contributed training samples
- `current_language` (str) - "en" or "da"

**Key Methods:**
- `predict_digit()` - Perform prediction in background thread
- `add_training_data(label)` - Store user drawing for training
- `train_new_model()` - Create and train new model
- `toggle_language()` - Switch between English and Danish
- `setup_bindings()` - Configure keyboard shortcuts

**Keyboard Shortcuts:**
| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Clear canvas |
| `Enter` / `Ctrl+X` | Predict digit |
| `Ctrl+V` | Add to training data |
| `Ctrl+N` | Train new model |

### Usage Example

```python
if __name__ == "__main__":
    app = DigitRecognitionApp()
    app.mainloop()
```

---

## 🧠 model_ensemble.py - Model Management

**Location:** `src/model_ensemble.py`  
**Purpose:** Load, manage, and use ensemble of CNN models  
**Lines:** ~191

### Key Class: `ModelEnsemble`

Manages multiple CNN models for prediction and training.

**Attributes:**
- `mnist_models` (list) - MNIST-trained models
- `emnist_models` (list) - EMNIST-trained models
- `model_paths` (list) - File paths to loaded models
- `models_dir` (str) - Directory containing .keras files

**Key Methods:**

#### `__init__()`
Initializes ensemble by scanning `models/` directory for .keras files.

```python
ensemble = ModelEnsemble()
# Automatically loads all models from models/ directory
```

#### `predict(image)`
Make prediction using ensemble averaging.

**Parameters:**
- `image` (np.array) - Shape (1, 28, 28, 1), normalized [0,1]

**Returns:**
- `np.array` - Shape (10,), confidence scores for digits 0-9

```python
predictions = ensemble.predict(preprocessed_image)
predicted_digit = np.argmax(predictions)
confidence = np.max(predictions)
```

#### `add_model(model_path, dataset_type='mnist')`
Add an existing model to the ensemble.

**Parameters:**
- `model_path` (str) - Path to .keras file
- `dataset_type` (str) - 'mnist' or 'emnist'

#### `create_new_model(images=None, labels=None, dataset_type='mnist')`
Train a new model with optional custom data.

**Parameters:**
- `images` (np.array, optional) - Training images, shape (N, 28, 28, 1)
- `labels` (np.array, optional) - Training labels, shape (N,)
- `dataset_type` (str) - 'mnist' or 'emnist'

**Behavior:**
- If no data provided, uses MNIST/EMNIST dataset
- Model saved as `model_{dataset_type}_{i}.keras`
- Automatically added to ensemble after training

```python
# Train with MNIST data
ensemble.create_new_model()

# Train with custom data
ensemble.create_new_model(
    images=custom_images,
    labels=custom_labels,
    dataset_type='mnist'
)
```

### Callbacks Used

- **EarlyStopping** - Stop training if validation loss doesn't improve for 5 epochs
- **ReduceLROnPlateau** - Reduce learning rate by 20% if loss plateaus

---

## 🖼️ widgets.py - UI Components

**Location:** `src/widgets.py`  
**Purpose:** Custom Tkinter widgets for drawing, display, and training  
**Lines:** ~202

### Key Classes

#### `DrawingCanvas(tk.Canvas)`
Interactive canvas for drawing digits.

**Attributes:**
- `width`, `height` - Canvas dimensions (default 280x280)
- `image` (PIL.Image) - Drawing stored as PIL image
- `line_width` - Brush width (default 12 pixels)

**Key Methods:**
- `paint(event)` - Handle mouse drawing
- `clear()` - Erase canvas
- `get_image()` - Return drawing as numpy array

```python
canvas = DrawingCanvas(root, width=280, height=280)
image_array = canvas.get_image()  # Get current drawing
canvas.clear()  # Clear drawing
```

#### `PredictionDisplay(tk.Frame)`
Display prediction results with confidence chart.

**Features:**
- Large prediction text
- Matplotlib bar chart showing confidence for all digits
- Real-time updates

#### `TrainingPanel(tk.Frame)`
Interface for collecting training data.

**Features:**
- Select correct digit if prediction wrong
- Add drawing to training dataset
- Train new model button

---

## 🤖 model.py - CNN Architecture

**Location:** `src/model.py`  
**Purpose:** Define CNN model architecture for digit recognition  
**Lines:** ~70

### Key Functions

#### `build_cnn_model() → Sequential`
Constructs and compiles CNN model architecture.

**Architecture:**
```
Input: (28, 28, 1)
  ↓
Conv2D(32, 3×3) + ReLU
  ↓
MaxPooling(2×2)
  ↓
Conv2D(64, 3×3) + ReLU
  ↓
MaxPooling(2×2)
  ↓
Flatten
  ↓
Dense(128) + ReLU
  ↓
Dropout(0.25)
  ↓
Dense(10) + Softmax
Output: (10,) - digit probabilities
```

**Compilation:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

```python
model = build_cnn_model()
# Ready to train or use for prediction
```

#### `load_or_train_model(model_path='cnn_model.keras') → Sequential`
Load existing model or train new one if file doesn't exist.

---

## 📊 data_utils.py - Data Handling

**Location:** `src/data_utils.py`  
**Purpose:** Load and preprocess datasets  
**Lines:** ~80

### Key Functions

#### `load_and_prepare_mnist(use_emnist=False) → tuple`
Load MNIST or EMNIST dataset with preprocessing.

**Returns:**
- `((x_train, y_train), (x_test, y_test))`

**Details:**
- Shapes: (N, 28, 28, 1)
- Values: Normalized to [0, 1]
- Falls back to MNIST if EMNIST fails

```python
(x_train, y_train), (x_test, y_test) = load_and_prepare_mnist()

# Or with EMNIST
(x_train, y_train), (x_test, y_test) = load_and_prepare_mnist(use_emnist=True)
```

#### `preprocess_image(image) → np.array`
Preprocess single image for model prediction.

**Input:** Any shape image (will be converted to grayscale)

**Processing:**
1. Convert to grayscale if needed
2. Resize to 28×28
3. Normalize to [0, 1]
4. Add batch dimension: shape (1, 28, 28, 1)

```python
processed = preprocess_image(raw_image)
prediction = model.predict(processed)
```

---

## 🎯 digit_preprocessing.py - Advanced Preprocessing

**Location:** `src/digit_preprocessing.py`  
**Purpose:** Sophisticated digit image preprocessing for canvas drawings  
**Lines:** ~35

### Key Functions

#### `preprocess_digit_image(img, preview_size=(140, 140)) → tuple`
Process canvas drawing for model input and GUI preview.

**Parameters:**
- `img` - Raw image from canvas (H×W or H×W×C)
- `preview_size` - Size for GUI preview display

**Returns:**
- `(processed_img, preview_img)` tuple
  - `processed_img`: shape (1, 28, 28, 1), float32, normalized
  - `preview_img`: PIL.ImageTk.PhotoImage for display

**Processing Pipeline:**
1. Binary thresholding (threshold=200)
2. Invert (black foreground, white background)
3. Contour detection
4. Bounding box extraction
5. Add padding (1/4 of digit size)
6. Crop to digit area
7. Create square canvas (padding with zeros)
8. Resize to 20×20
9. Place in center of 28×28 MNIST format
10. Normalize to [0, 1]

**Example:**
```python
canvas_image = canvas.get_image()
processed, preview = preprocess_digit_image(canvas_image)

# Use processed for model prediction
predictions = model.predict(processed)

# Display preview in GUI
label.config(image=preview)
label.image = preview  # Keep reference
```

---

## 📝 Utility Scripts

### `create_models.py`
Create initial MNIST and EMNIST models.

**Usage:**
```bash
python create_models.py
```

### `test_model_load.py`
Test model loading functionality.

### `test_models.py`
Test model predictions.

### `training.py`
Custom training script.

---

## Import Dependencies Summary

| Module | Imports |
|--------|---------|
| gui.py | tkinter, threading, numpy, PIL, tensorflow, model_ensemble, widgets |
| model_ensemble.py | numpy, os, tensorflow, model, data_utils |
| widgets.py | tkinter, numpy, PIL, matplotlib, digit_preprocessing |
| model.py | tensorflow.keras |
| data_utils.py | cv2, numpy, tensorflow, tensorflow_datasets |
| digit_preprocessing.py | numpy, cv2, PIL |

---

See also: [Architecture Overview](Architecture.md), [GUI Components](GUI-Components.md), [API Documentation](API-Documentation.md)
