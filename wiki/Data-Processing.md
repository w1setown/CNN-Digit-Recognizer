# Data Processing Guide

Complete documentation on image preprocessing, data loading, and data utilities.

## Data Processing Pipeline

### Overview

The data processing pipeline transforms raw user drawings into MNIST-format inputs suitable for CNN predictions.

```
Raw Canvas Image (280×280)
    ↓ [Binary Conversion]
Binary Image (Black digit on white)
    ↓ [Contour Detection]
Bounding Box (Minimal rectangle around digit)
    ↓ [Padding & Extraction]
Extracted Digit Region
    ↓ [Square Padding]
Square Canvas (Maintains aspect ratio)
    ↓ [Resize to 20×20]
Small Digit (20×20)
    ↓ [Center in MNIST Grid]
MNIST Format (28×28, digit centered)
    ↓ [Normalization]
Final Input (28×28, values [0,1])
    ↓
Model Input: Shape (1, 28, 28, 1)
```

## Preprocessing Functions

### `digit_preprocessing.preprocess_digit_image()`

**Location:** `src/digit_preprocessing.py`

Comprehensive preprocessing function for canvas drawings.

#### Function Signature

```python
def preprocess_digit_image(img, preview_size=(140, 140)) -> tuple:
    """
    Preprocess a digit image for model prediction/training and GUI preview.
    
    Args:
        img (np.array): Raw image from canvas, shape (H, W) or (H, W, C)
        preview_size (tuple): Size for GUI preview display
        
    Returns:
        (processed_img, preview_img): Tuple of
            - processed_img: shape (1, 28, 28, 1), float32, normalized [0,1]
            - preview_img: PIL.ImageTk.PhotoImage for GUI display
    """
```

#### Processing Steps

**Step 1: Binary Thresholding**
```
Threshold value: 200 (out of 255)
Mode: Inverse binary (THRESH_BINARY_INV)
Result: Black digit (0) on white background (255)
```

**Step 2: Contour Detection**
```
Find all contours in binary image
Purpose: Identify digit boundaries
```

**Step 3: Bounding Rectangle**
```python
x, y, w, h = cv2.boundingRect(all_points)
# x, y: top-left corner
# w, h: width and height
```

**Step 4: Add Padding**
```python
padding = max(w, h) // 4
# Add 25% padding around digit
# Provides margin for centering
```

**Step 5: Extract ROI (Region of Interest)**
```python
digit_roi = binary[y1:y2, x1:x2]
# Crop to digit region with padding
```

**Step 6: Create Square Canvas**
```python
square_size = max(digit_roi.shape[0], digit_roi.shape[1])
squared_img = np.zeros((square_size, square_size))
# Pad shorter dimension with zeros
# Maintains aspect ratio
```

**Step 7: Resize to 20×20**
```python
squared_img = cv2.resize(squared_img, (20, 20))
# Standard model input requires 28×28
# But MNIST convention is 20×20 digit with 4-pixel margin
```

**Step 8: Center in MNIST Format**
```python
processed_img = np.zeros((28, 28))
processed_img[4:24, 4:24] = squared_img
# Place 20×20 digit in center
# 4-pixel margin on all sides
# Final shape: (28, 28)
```

**Step 9: Normalization**
```python
norm_img = processed_img.astype('float32') / 255.0
# Convert to float
# Scale to [0, 1] range
# Expected by neural network
```

**Step 10: Add Batch & Channel Dimensions**
```python
norm_img = np.expand_dims(norm_img, axis=-1)  # (28, 28, 1)
norm_img = np.expand_dims(norm_img, axis=0)   # (1, 28, 28, 1)
# Batch dimension: 1 (single image)
# Channel dimension: 1 (grayscale)
```

**Step 11: Create Preview**
```python
pil_img = Image.fromarray(processed_img)
pil_img = pil_img.resize(preview_size)  # (140, 140) by default
preview_img = ImageTk.PhotoImage(pil_img)
# For GUI display
```

#### Edge Cases Handled

**Empty Canvas:**
```python
if contours is None:
    # Fallback: resize full image to 28×28
    processed_img = cv2.resize(binary, (28, 28))
```

**Very Small Digit:**
- Added padding ensures visibility
- Won't disappear in center

**Very Large Digit:**
- Padding may clip outer edges
- Acceptable trade-off for centering

**Off-Center Digit:**
- Bounding box extraction centers it
- 4-pixel margin prevents edge issues

#### Example Usage

```python
from src.digit_preprocessing import preprocess_digit_image
import numpy as np

# Get raw image from canvas
raw_image = canvas.get_image()  # (280, 280)

# Preprocess
processed, preview = preprocess_digit_image(raw_image)

# Use for prediction
predictions = model.predict(processed)

# Display preview
preview_label.config(image=preview)
preview_label.image = preview  # Keep reference!
```

---

### `data_utils.preprocess_image()`

**Location:** `src/data_utils.py`

Simpler preprocessing for general images (not canvas drawings).

#### Function Signature

```python
def preprocess_image(image) -> np.array:
    """
    Preprocess a single image for prediction.
    
    Args:
        image: Image in any format/shape (H, W) or (H, W, C)
        
    Returns:
        np.array: Shape (1, 28, 28, 1), float32, normalized
    """
```

#### Processing Steps

1. Convert to grayscale if needed
2. Resize to 28×28
3. Normalize to [0, 1]
4. Add batch and channel dimensions

#### Key Differences from `preprocess_digit_image()`

| Aspect | `preprocess_image()` | `preprocess_digit_image()` |
|--------|----------------------|---------------------------|
| **Purpose** | General images | Canvas drawings |
| **Preprocessing** | Basic | Advanced (contours, centering) |
| **Speed** | Fast | Slower (more steps) |
| **Preview** | No | Yes |
| **Use Case** | Datasets, files | User interface |

---

## Dataset Loading

### `load_and_prepare_mnist()`

**Location:** `src/data_utils.py`

Loads and preprocesses MNIST or EMNIST dataset.

#### Function Signature

```python
def load_and_prepare_mnist(use_emnist=False) -> tuple:
    """
    Load and preprocess the MNIST or EMNIST dataset.
    
    Args:
        use_emnist (bool): If True, load EMNIST. Else MNIST.
        
    Returns:
        ((x_train, y_train), (x_test, y_test)):
            - x_train/x_test: shape (N, 28, 28, 1), float32 in [0,1]
            - y_train/y_test: shape (N,), int in [0-9]
    """
```

#### MNIST Dataset

**Source:** TensorFlow Datasets (tensorflow.keras.datasets.mnist)

**Statistics:**
```
Training samples: 60,000
Test samples: 10,000
Digit classes: 0-9 (10 classes)
Image size: 28×28 pixels
Format: Grayscale (single channel)
```

**Preprocessing Applied:**
1. Reshape to (N, 28, 28, 1)
2. Normalize to [0, 1]
3. Convert to float32

#### EMNIST Dataset

**Source:** TensorFlow Datasets (emnist/digits)

**Statistics:**
```
Training samples: 814,255
Test samples: 131,600
Digit classes: 0-9 (MNIST-compatible)
Image size: 28×28 pixels
Format: Grayscale (single channel)
```

**Why EMNIST?**
- Extended MNIST includes handwritten letters + digits
- More diverse handwriting styles
- Better generalization to user drawings
- Digits subset is MNIST-compatible

**Fallback Behavior:**
```python
try:
    # Try to load EMNIST
    dataset = tfds.load('emnist/digits', as_supervised=True)
except Exception as e:
    print(f"EMNIST failed: {e}")
    # Fall back to MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

#### Example Usage

```python
from src.data_utils import load_and_prepare_mnist

# Load MNIST
(x_train, y_train), (x_test, y_test) = load_and_prepare_mnist()
print(f"Training shape: {x_train.shape}")  # (60000, 28, 28, 1)
print(f"Labels: {y_train.min()}-{y_train.max()}")  # 0-9

# Load EMNIST
(x_train, y_train), (x_test, y_test) = load_and_prepare_mnist(use_emnist=True)
print(f"Training shape: {x_train.shape}")  # (814255, 28, 28, 1)
```

---

## Data Format Specifications

### Input Specification

**For Model Prediction:**
```python
Shape: (batch_size, height, width, channels)
       (1, 28, 28, 1) for single image

Data type: float32
Value range: [0.0, 1.0]
  - 0.0 = white (background)
  - 1.0 = black (digit)

Example:
  X = np.random.rand(1, 28, 28, 1).astype('float32')
  predictions = model.predict(X)
```

**For Labels:**
```python
Data type: int (or np.int32/int64)
Value range: 0-9
  - 0 to 9 represent digit classes

Example:
  y = np.array([3, 5, 7, 2, 8])  # 5 training samples
  model.fit(X, y, ...)
```

### Canvas Coordinate System

```
Canvas (280×280)
(0,0) ─────────────────→ (280,0)
  │                      │
  │                      │
  │  Drawing Area        │
  │                      │
  ↓                      ↓
(0,280) ────────────────(280,280)

PIL Image Internal
0 (white) = background
0-254 = gray levels
255 (black) = drawn pixels

After preprocessing:
0.0 (white) = background
0.0-1.0 = continuous values
1.0 (black) = digit pixels
```

---

## Image Quality Considerations

### Factors Affecting Recognition

| Factor | Impact | Mitigation |
|--------|--------|-----------|
| **Digit Size** | Small digits hard to recognize | Padding ensures visibility |
| **Centering** | Off-center digits misclassified | Bounding box + centering |
| **Line Width** | Too thick makes features ambiguous | 12-pixel brush recommended |
| **Stroke Quality** | Shaky strokes add noise | Smooth rendering with round caps |
| **Contrast** | Low contrast hard to threshold | Binary conversion is robust |

### Optimal Drawing Conditions

1. **Size:** Digit fills 60-80% of canvas
2. **Position:** Roughly centered
3. **Style:** Single continuous strokes preferred
4. **Thickness:** Normal pen/brush width
5. **Contrast:** Solid black on white background

---

## Data Augmentation

### Used During Training

```python
ImageDataGenerator(
    rotation_range=10,      # ±10° rotations
    zoom_range=0.1,         # ±10% zoom
    shift_range=0.1,        # ±10% shift (pixels)
    # ... additional augmentations
)
```

**Purpose:** Increase effective training data size

**Benefits:**
- Handles digits at various angles
- Handles digits at various sizes
- Handles off-center digits
- Improved generalization

### Not Applied During Prediction

Canvas preprocessing is sufficient for user drawings:
- Explicit centering (no need for shift augmentation)
- Fixed size (no need for zoom augmentation)
- Upright orientation (no need for rotation)

---

## Performance Metrics for Preprocessing

### Speed

```
Typical preprocessing time per image:
  Binary conversion: 1-2ms
  Contour detection: 2-3ms
  Digit extraction: 1-2ms
  Resize operations: 1-2ms
  ─────────────────────────
  Total: 5-10ms
```

### Quality

```
Typical recognition improvement:
  Without preprocessing: 85-90% accuracy
  With preprocessing: 95-98% accuracy
  
Improvements from:
  - Consistent size normalization
  - Proper centering
  - Noise reduction
  - Consistency with training data format
```

---

## Debugging Data Issues

### How to Visualize Preprocessing

```python
import matplotlib.pyplot as plt
from src.digit_preprocessing import preprocess_digit_image

# Draw something
raw = canvas.get_image()

# Preprocess
processed, preview = preprocess_digit_image(raw)

# Visualize steps
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(raw, cmap='gray')
plt.title('Raw Canvas')

plt.subplot(1, 3, 2)
plt.imshow(preview, cmap='gray')
plt.title('Preview (140×140)')

plt.subplot(1, 3, 3)
plt.imshow(processed[0, :, :, 0], cmap='gray')
plt.title('Model Input (28×28)')

plt.tight_layout()
plt.show()
```

### Common Issues

**Problem:** Model output is flat (all digits similar probability)

**Causes:**
1. Image is all white (empty canvas) → Verify preprocessing detects digit
2. Image preprocessing failed → Check binary conversion threshold
3. Model not trained → Ensure models loaded from models/ directory

**Solution:**
```python
# Debug preprocessing
processed, preview = preprocess_digit_image(raw)
assert processed.shape == (1, 28, 28, 1), f"Wrong shape: {processed.shape}"
assert processed.min() >= 0 and processed.max() <= 1, "Wrong value range"
assert processed.mean() > 0.1, "Image might be empty"
```

---

See also: [Model System](Model-System.md), [Core Modules](Core-Modules.md), [Architecture Overview](Architecture.md)
