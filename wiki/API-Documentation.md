# API Documentation

Complete API reference for all functions and classes in the CNN Digit Recognizer.

## Core Classes

### DigitRecognitionApp (gui.py)

Main application window class.

**Inherits from:** `tk.Tk`

#### Constructor

```python
class DigitRecognitionApp(tk.Tk):
    def __init__(self)
```

**Behavior:**
- Creates main window
- Initializes UI components
- Loads models
- Sets up event bindings

#### Key Methods

##### `predict_digit()`

```python
def predict_digit(self) -> None
```

**Purpose:** Get prediction from ensemble

**Behavior:**
1. Extracts drawing from canvas
2. Preprocesses image
3. Runs through model ensemble
4. Updates display
5. Runs in background thread

**Called by:** `Enter` key, "Guess my digit!" button

---

##### `add_training_data(label: int) -> None`

```python
def add_training_data(label: int) -> None
```

**Parameters:**
- `label` (int) - Correct digit (0-9)

**Behavior:**
- Stores current canvas image + label
- Adds to training dataset
- Updates counter display
- Shows confirmation message

---

##### `train_new_model() -> None`

```python
def train_new_model(self) -> None
```

**Purpose:** Train new model with collected data

**Behavior:**
1. Validates sufficient data collected
2. Spawns training thread
3. Shows progress dialog
4. Saves new model
5. Adds to ensemble
6. Updates display

**Called by:** `Ctrl+N`, "Teach the AI" button

---

##### `toggle_language() -> None`

```python
def toggle_language(self) -> None
```

**Purpose:** Switch between English and Danish

**Updates:**
- Button labels
- Help text
- Error messages
- UI strings

---

### ModelEnsemble (model_ensemble.py)

Manages multiple CNN models for predictions.

**Location:** `src/model_ensemble.py`

#### Constructor

```python
class ModelEnsemble:
    def __init__(self)
```

**Behavior:**
- Scans `models/` directory
- Loads all `.keras` files
- Categorizes as MNIST/EMNIST
- Creates fallback models if none exist

**Attributes After Init:**
```python
self.mnist_models : list[tf.keras.Model]
self.emnist_models : list[tf.keras.Model]
self.model_paths : list[str]
self.models_dir : str
```

---

#### `predict(image: np.ndarray) -> np.ndarray`

```python
def predict(self, image: np.ndarray) -> np.ndarray
```

**Parameters:**
- `image` (np.ndarray) - Shape (1, 28, 28, 1), dtype float32, values [0, 1]

**Returns:**
- np.ndarray - Shape (10,), softmax probabilities for digits 0-9

**Process:**
1. Run image through all MNIST models
2. Run image through all EMNIST models
3. Collect all predictions
4. Average predictions
5. Return averaged probabilities

**Example:**
```python
ensemble = ModelEnsemble()
predictions = ensemble.predict(preprocessed_image)
digit = np.argmax(predictions)
confidence = predictions[digit]
```

---

#### `add_model(model_path: str, dataset_type: str = 'mnist') -> None`

```python
def add_model(self, model_path: str, dataset_type: str = 'mnist') -> None
```

**Parameters:**
- `model_path` (str) - Path to .keras file
- `dataset_type` (str) - 'mnist' or 'emnist'

**Behavior:**
- Loads model from file
- Adds to appropriate list
- Stores path reference

---

#### `create_new_model(images=None, labels=None, dataset_type='mnist', base_path=None) -> None`

```python
def create_new_model(
    self,
    images: np.ndarray = None,
    labels: np.ndarray = None,
    dataset_type: str = 'mnist',
    base_path: str = None
) -> None
```

**Parameters:**
- `images` (np.ndarray, optional) - Training images, shape (N, 28, 28, 1)
- `labels` (np.ndarray, optional) - Training labels, shape (N,), values 0-9
- `dataset_type` (str) - 'mnist' or 'emnist' (used if no data provided)
- `base_path` (str, optional) - Path for saving model

**Behavior:**
1. If no data provided, loads MNIST/EMNIST
2. Builds new CNN model
3. Applies data augmentation
4. Trains with callbacks (EarlyStopping, ReduceLROnPlateau)
5. Saves model to `models/` directory
6. Automatically adds to ensemble
7. Prints training progress

**Example:**
```python
# Using MNIST dataset
ensemble.create_new_model(dataset_type='mnist')

# Using custom data
ensemble.create_new_model(
    images=user_images,
    labels=user_labels,
    dataset_type='mnist'
)
```

---

## Widget Classes

### DrawingCanvas (widgets.py)

Interactive canvas for drawing digits.

**Inherits from:** `tk.Canvas`

#### Constructor

```python
class DrawingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs)
```

**Parameters:**
- `parent` - Parent widget
- `width` - Canvas width (default 280)
- `height` - Canvas height (default 280)
- Other `tk.Canvas` kwargs

**Attributes:**
```python
self.image : PIL.Image      # Drawing storage
self.draw : PIL.ImageDraw   # Drawing interface
self.line_width : int       # Brush width (default 12)
self.width : int            # Canvas width
self.height : int           # Canvas height
```

#### Methods

##### `get_image() -> np.ndarray`

```python
def get_image(self) -> np.ndarray
```

**Returns:** Canvas drawing as numpy array

**Example:**
```python
image = canvas.get_image()  # Shape (280, 280)
```

---

##### `clear() -> None`

```python
def clear(self) -> None
```

**Purpose:** Erase canvas and reset state

**Behavior:**
- Clears Tkinter canvas
- Resets PIL image (white)
- Resets drawing state variables

---

##### `paint(event) -> None`

```python
def paint(self, event) -> None
```

**Purpose:** Handle mouse drawing (internal use)

**Parameters:**
- `event` - Tkinter mouse event

---

### PredictionDisplay (widgets.py)

Displays model predictions with confidence chart.

**Inherits from:** `tk.Frame`

#### Methods

##### `update_prediction(predictions: np.ndarray, digit: int) -> None`

```python
def update_prediction(
    self,
    predictions: np.ndarray,
    digit: int
) -> None
```

**Parameters:**
- `predictions` (np.ndarray) - Shape (10,), softmax probabilities
- `digit` (int) - Predicted digit (0-9)

**Behavior:**
- Updates large digit display
- Updates confidence percentage
- Redraws bar chart
- Colors highest bar green

---

### TrainingPanel (widgets.py)

Interface for collecting training data.

**Inherits from:** `tk.Frame`

#### Methods

##### `get_selected_digit() -> int`

```python
def get_selected_digit(self) -> int
```

**Returns:** Currently selected digit (0-9)

---

## Core Functions

### Preprocessing

#### `preprocess_digit_image(img, preview_size=(140, 140)) -> tuple`

**Location:** `src/digit_preprocessing.py`

```python
def preprocess_digit_image(
    img: np.ndarray,
    preview_size: tuple = (140, 140)
) -> tuple
```

**Parameters:**
- `img` (np.ndarray) - Canvas image, any shape
- `preview_size` (tuple) - Display preview size

**Returns:**
- `(processed_img, preview_img)` tuple
  - `processed_img` (np.ndarray) - Shape (1, 28, 28, 1), float32, [0, 1]
  - `preview_img` (PIL.ImageTk.PhotoImage) - For GUI display

**Process:**
1. Binary thresholding (threshold=200)
2. Contour detection
3. Bounding box extraction
4. Padding and centering
5. Resize to 28×28 MNIST format
6. Normalization
7. Create preview

**Example:**
```python
from src.digit_preprocessing import preprocess_digit_image

processed, preview = preprocess_digit_image(raw_image)

# Use for prediction
predictions = model.predict(processed)

# Display preview
label.config(image=preview)
```

---

#### `preprocess_image(image) -> np.ndarray`

**Location:** `src/data_utils.py`

```python
def preprocess_image(image: np.ndarray) -> np.ndarray
```

**Parameters:**
- `image` (np.ndarray) - Image in any format

**Returns:**
- np.ndarray - Shape (1, 28, 28, 1), float32, [0, 1]

**Process:**
1. Convert to grayscale
2. Resize to 28×28
3. Normalize to [0, 1]
4. Add batch and channel dimensions

---

### Data Loading

#### `load_and_prepare_mnist(use_emnist=False) -> tuple`

**Location:** `src/data_utils.py`

```python
def load_and_prepare_mnist(
    use_emnist: bool = False
) -> tuple
```

**Parameters:**
- `use_emnist` (bool) - Load EMNIST if True, MNIST if False

**Returns:**
- `((x_train, y_train), (x_test, y_test))` tuple
  - `x_train`, `x_test` (np.ndarray) - Shape (N, 28, 28, 1), float32, [0, 1]
  - `y_train`, `y_test` (np.ndarray) - Shape (N,), int 0-9

**Behavior:**
- Downloads dataset on first use
- Caches for subsequent calls
- Falls back to MNIST if EMNIST fails
- Preprocesses (reshape, normalize)

**Example:**
```python
from src.data_utils import load_and_prepare_mnist

# Load MNIST
(x_train, y_train), (x_test, y_test) = load_and_prepare_mnist()

# Load EMNIST
(x_train, y_train), (x_test, y_test) = load_and_prepare_mnist(use_emnist=True)
```

---

### Model Building

#### `build_cnn_model() -> tf.keras.Sequential`

**Location:** `src/model.py`

```python
def build_cnn_model() -> tf.keras.Sequential
```

**Returns:** Compiled Keras Sequential model

**Architecture:**
```
Input: (28, 28, 1)
  ↓
Conv2D(32, 3×3, ReLU) → MaxPooling(2×2)
  ↓
Conv2D(64, 3×3, ReLU) → MaxPooling(2×2)
  ↓
Flatten
  ↓
Dense(128, ReLU) → Dropout(0.25)
  ↓
Dense(10, Softmax)
Output: (10,)
```

**Compilation:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

**Example:**
```python
from src.model import build_cnn_model

model = build_cnn_model()
model.fit(x_train, y_train, ...)
predictions = model.predict(x_test)
```

---

#### `load_or_train_model(model_path='cnn_model.keras') -> tf.keras.Sequential`

**Location:** `src/model.py`

```python
def load_or_train_model(
    model_path: str = 'cnn_model.keras'
) -> tf.keras.Sequential
```

**Parameters:**
- `model_path` (str) - Path to model file

**Returns:** Loaded or newly trained model

**Behavior:**
- If file exists: Load model
- If file doesn't exist: Build and train new model
- Trains on MNIST dataset if needed

---

## Type Hints Reference

### Input Types

```python
# Image arrays
np.ndarray : Shape (H, W) or (H, W, C), dtype any numeric type

# Model inputs
np.ndarray : Shape (1, 28, 28, 1), dtype float32, values [0, 1]

# Labels
np.ndarray : Shape (N,), dtype int, values 0-9

# Predictions
np.ndarray : Shape (10,), dtype float32, values [0, 1], sum = 1.0
```

### Return Types

```python
# Predictions
np.ndarray : Shape (10,), dtype float32

# Images
np.ndarray : Various shapes depending on context
PIL.Image : PIL image object
PIL.ImageTk.PhotoImage : Tkinter-compatible image

# Models
tf.keras.Sequential : Compiled Keras model
```

---

## Error Handling

### Common Exceptions

**FileNotFoundError**
```python
# Raised when model file not found
# Check file path and models/ directory
```

**ValueError**
```python
# Raised when image shape incorrect
# Ensure preprocessing produces (1, 28, 28, 1)
```

**TensorFlowError**
```python
# Raised during model operations
# Check TensorFlow version compatibility
```

**Memory Error**
```python
# Raised if insufficient RAM
# Reduce model count or increase swap
```

---

## Performance Characteristics

### Latency

```python
# Typical timings
preprocess_digit_image():  5-10ms
model.predict() (single):  10-30ms
ensemble.predict():        30-50ms
training (per epoch):      1-5 seconds
```

### Memory Usage

```python
# Approximate memory consumption
Single model:              50-100 MB
3-model ensemble:          150-300 MB
Preprocessing image:       1-2 MB
Training data (100 samples): 5-10 MB
```

---

## Callback Parameters

### EarlyStopping

```python
EarlyStopping(
    monitor='val_loss',        # Metric to watch
    patience=5,                # Epochs without improvement
    restore_best_weights=True  # Restore best weights
)
```

### ReduceLROnPlateau

```python
ReduceLROnPlateau(
    monitor='val_loss',    # Metric to watch
    factor=0.2,            # Multiply LR by 0.2
    patience=3,            # Epochs without improvement
    min_lr=0.0001          # Minimum learning rate
)
```

---

See also: [Core Modules](Core-Modules.md), [Model System](Model-System.md), [Data Processing](Data-Processing.md)
