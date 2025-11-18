# Model System Documentation

Comprehensive guide to the CNN model architecture, ensemble system, and training pipeline.

## CNN Architecture

### Network Structure

The project uses a **Convolutional Neural Network (CNN)** optimized for handwritten digit recognition on 28×28 MNIST-format images.

#### Model Layers

```
Input Layer
  Shape: (28, 28, 1) - Grayscale 28×28 pixel images
  
  ↓
  
Block 1 - Feature Extraction
  Conv2D(filters=32, kernel_size=3×3, activation='relu')
    → 32 feature maps
  MaxPooling2D(pool_size=2×2)
    → Spatial reduction: (14, 14)
  
  ↓
  
Block 2 - Advanced Features
  Conv2D(filters=64, kernel_size=3×3, activation='relu')
    → 64 feature maps
  MaxPooling2D(pool_size=2×2)
    → Spatial reduction: (7, 7)
  
  ↓
  
Flatten Layer
  Convert 2D feature maps to 1D vector
  Size: 64 × 7 × 7 = 3136 values
  
  ↓
  
Dense Block - Classification
  Dense(128, activation='relu')
    → 128 neurons
  Dropout(0.25)
    → Drop 25% of connections to prevent overfitting
  
  ↓
  
Output Layer
  Dense(10, activation='softmax')
    → 10 classes (digits 0-9)
    → Probabilities sum to 1.0
```

#### Model Diagram

```
Input (28×28×1)
    ↓
[Conv2D 32] → [ReLU] → [MaxPool 2×2]  →  (14×14×32)
    ↓
[Conv2D 64] → [ReLU] → [MaxPool 2×2]  →  (7×7×64)
    ↓
[Flatten]                              →  (3136,)
    ↓
[Dense 128] → [ReLU] → [Dropout 0.25]
    ↓
[Dense 10] → [Softmax]                 →  (10,) probabilities
```

### Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| **Conv Filters** | 32 → 64 | Progressive feature complexity |
| **Kernel Size** | 3×3 | Good balance of receptive field and parameter count |
| **Pooling** | Max, 2×2 | Translation invariance, dimensionality reduction |
| **Activation** | ReLU | Fast, non-linear, prevents vanishing gradients |
| **Dropout** | 0.25 | Moderate regularization without over-constraining |
| **Output** | Softmax | Multi-class probability distribution |

### Computational Summary

```
Total Parameters: ~50,000 (varies with exact layer configs)
Memory per Prediction: ~1-2 MB
Inference Time: ~10-50ms per image
Suitable for: Real-time interactive applications
```

---

## Compilation Configuration

The model is compiled with:

```python
model.compile(
    optimizer='adam',           # Adaptive learning rate
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']        # Track accuracy during training
)
```

### Why These Choices?

**Adam Optimizer:**
- Adaptive learning rates per parameter
- Works well with mini-batch training
- Converges faster than standard SGD

**Sparse Categorical Crossentropy:**
- For integer labels (0-9) instead of one-hot encoded
- Mathematically identical to categorical crossentropy but more efficient
- Standard choice for multi-class classification

**Accuracy Metric:**
- Simple to interpret
- Direct measure of correct classifications
- Suitable for balanced dataset (MNIST)

---

## Model Ensemble System

### Why Ensemble?

Single model limitations:
- Individual biases or failure modes
- Overfitting to training data quirks
- Sensitivity to initialization

Ensemble benefits:
- **Robustness:** Average multiple predictions
- **Accuracy:** Combined models often better than individual
- **Confidence:** Multiple independent predictions increase reliability

### Ensemble Architecture

```
User Input Image
    ↓
[Preprocessing]
    ↓
    ├─→ Model 1 (mnist_0)  → [0.05, 0.1, 0.8, ...]
    ├─→ Model 2 (mnist_1)  → [0.08, 0.12, 0.75, ...]
    ├─→ Model 3 (emnist_0) → [0.06, 0.09, 0.82, ...]
    └─→ Model N (...)      → [...]
    ↓
[Average Predictions]
    → [0.063, 0.103, 0.790, ...]
    ↓
[Argmax & Display]
    → Predicted digit: 2
    → Confidence: 79.0%
```

### ModelEnsemble Class

**Location:** `src/model_ensemble.py`

#### Initialization

```python
ensemble = ModelEnsemble()
```

**Behavior:**
1. Scans `models/` directory for `.keras` files
2. Loads each model with error handling
3. Categorizes as MNIST or EMNIST based on filename
4. Reports loading status

#### Model Categories

**MNIST Models:**
- Filename pattern: `model_mnist_*.keras`
- Trained on original MNIST dataset
- 70,000 training samples (60k train + 10k test)

**EMNIST Models:**
- Filename pattern: `model_emnist_*.keras`
- Trained on extended MNIST (handwritten letters + digits)
- 814,255 training samples
- More robust to varied handwriting

#### Prediction Method

```python
predictions = ensemble.predict(image)
```

**Input:**
- `image`: numpy array, shape (1, 28, 28, 1), dtype float32, values [0, 1]

**Process:**
1. Pass image through all MNIST models
2. Pass image through all EMNIST models
3. Collect all predictions
4. Average along axis 0
5. Return averaged probabilities

**Output:**
- numpy array, shape (10,)
- Values sum to 1.0
- Index = digit class, value = confidence

**Example:**
```python
predictions = ensemble.predict(image)
# Output: [0.02, 0.01, 0.85, 0.05, 0.02, 0.01, 0.02, 0.01, 0.01, 0.0]
# → Predicted digit: 2, Confidence: 85%
```

---

## Training Pipeline

### Creating New Models

#### Automatic Training (Using MNIST/EMNIST)

```python
ensemble.create_new_model(dataset_type='mnist')
```

**What Happens:**
1. Load MNIST training data (60,000 samples)
2. Build fresh CNN model
3. Train for up to 50 epochs
4. Apply callbacks (EarlyStopping, ReduceLROnPlateau)
5. Save as `model_mnist_{i}.keras` where i = next index
6. Add to ensemble automatically

#### Custom Training (User Contributions)

```python
# Collect user drawings
training_images = []  # List of preprocessed images
training_labels = []  # List of correct digit labels

# Then train
ensemble.create_new_model(
    images=np.array(training_images),
    labels=np.array(training_labels),
    dataset_type='mnist'
)
```

### Training Callbacks

#### Early Stopping

```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

**Purpose:** Prevent overfitting

**Behavior:**
- Monitor validation loss each epoch
- If no improvement for 5 consecutive epochs
- Stop training and restore best weights
- Avoid wasting time and overfitting

#### Reduce Learning Rate on Plateau

```python
ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
```

**Purpose:** Fine-tune learning near optimum

**Behavior:**
- If validation loss plateaus for 3 epochs
- Reduce learning rate by 80% (multiply by 0.2)
- Minimum learning rate: 0.0001
- Helps escape local minima and refine weights

### Training Data Augmentation

```python
ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    shift_range=0.1,
    # ... other augmentations
)
```

**Purpose:** Increase effective training data size

**Techniques:**
- Random rotations (±10°) - Handle varied orientations
- Random zoom (±10%) - Handle different digit sizes
- Random shifts (±10% pixels) - Handle off-center digits

---

## Model Files & Storage

### File Format

**Format:** Keras H5/SavedModel (.keras format)

**Advantages:**
- Stores weights and architecture together
- Serializes optimizer state for resuming training
- Platform-independent
- Good compression

### File Naming Convention

```
model_{dataset}_{index}.keras

Examples:
  model_mnist_0.keras
  model_mnist_1.keras
  model_emnist_0.keras
  model_emnist_1.keras
```

**Convention Benefits:**
- Easy to identify training dataset
- Sequential indexing for multiple models
- Automatic discovery by ModelEnsemble

### Directory Structure

```
CNN-Digit-Recognizer/
├── models/
│   ├── model_mnist_0.keras       (60 MB typical)
│   ├── model_mnist_1.keras       (60 MB typical)
│   ├── model_emnist_0.keras      (60 MB typical)
│   └── ...
```

### File Size Considerations

Typical model size: **50-100 MB** per .keras file

**Breakdown:**
- Weights: ~40 MB (50k parameters × 8 bytes float64)
- Metadata: ~5 MB
- Compression: ~30-50% reduction

---

## Model Performance Metrics

### Expected Accuracy

| Dataset | Single Model | Ensemble (3 models) |
|---------|-------------|-------------------|
| MNIST Test Set | 98-99% | 99-99.5% |
| MNIST Validation | 97-98% | 98.5-99% |
| User Drawings | 92-96% | 95-97% |

### Inference Performance

| Metric | Value |
|--------|-------|
| Single Model Prediction | 10-30ms |
| Ensemble (3 models) | 30-50ms |
| Preprocessing | 5-10ms |
| Total Latency | 35-60ms |

### Why Ensemble Improves Accuracy

1. **Variance Reduction:** Multiple independent estimates reduce noise
2. **Bias Cancellation:** Different models have different biases that cancel
3. **Complementary Learning:** Different initializations learn different features
4. **Robustness:** Handles edge cases better than single model

---

## Adding Custom Models

### Option 1: Train & Auto-Add

```python
# Programmatic
ensemble.create_new_model(dataset_type='mnist')
# Model automatically saved and added to ensemble
```

### Option 2: External Model

```bash
# 1. Train externally with your own code
# 2. Save as .keras file
# 3. Place in models/ directory

# On next app start, automatically loaded
```

### Option 3: Transfer Learning

```python
# Load pre-trained model
base_model = tf.keras.models.load_model('existing.keras')

# Modify for new task
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(...)
# Train with new data
```

---

## Troubleshooting Model Issues

### Models Not Loading

**Symptom:** "Error loading model_mnist_0.keras"

**Solutions:**
1. Verify file is valid: `python -c "import tensorflow as tf; tf.keras.models.load_model('models/model_mnist_0.keras')"`
2. Check file isn't corrupted
3. Ensure TensorFlow version matches

### Poor Prediction Accuracy

**Symptom:** Wrong digit predictions

**Causes & Fixes:**
1. Digit poorly drawn → Ensure clear, centered digit
2. Model untrained → Run `create_models.py`
3. Wrong preprocessing → Check preprocessing pipeline
4. Model overfitted → Add more diverse training data

### Slow Predictions

**Symptom:** 1+ second prediction latency

**Causes & Fixes:**
1. Too many models loaded → Remove unnecessary models
2. System CPU limited → Close other applications
3. First prediction slower → Subsequent predictions cached

---

## Future Improvements

### Potential Enhancements

1. **Confidence Calibration** - Map raw probabilities to accurate confidence
2. **Rejection Option** - Refuse low-confidence predictions
3. **Uncertainty Quantification** - Predict prediction uncertainty
4. **Model Compression** - Quantization to reduce file size
5. **GPU Acceleration** - Leverage CUDA for faster inference
6. **Model Interpretability** - Visualize learned features (t-SNE, CAM)

---

See also: [Core Modules](Core-Modules.md), [Data Processing](Data-Processing.md), [Architecture Overview](Architecture.md)
