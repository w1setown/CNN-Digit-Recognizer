# Configuration Guide

Guide to configurable settings and customization options in the CNN Digit Recognizer.

## Application Settings

### GUI Configuration

#### Canvas Size

**Default:** 280×280 pixels

**To change:** Edit `src/gui.py`

```python
# In DigitRecognitionApp.__init__()
canvas_width = 280
canvas_height = 280

self.canvas = DrawingCanvas(
    canvas_frame,
    width=canvas_width,
    height=canvas_height
)
```

**Considerations:**
- Affects preprocessing (resizes to 28×28)
- Larger canvas: Better drawing control, more detail
- Smaller canvas: Faster processing
- Default 280×280 matches MNIST aspect ratio well

#### Canvas Line Width

**Default:** 12 pixels

**To change:** Edit `src/widgets.py`

```python
# In DrawingCanvas.__init__()
self.line_width = 12
```

**Considerations:**
- Larger width: Easier to see strokes
- Smaller width: More detail, harder control
- 12 pixels optimal for most users
- 8-15 pixels reasonable range

#### Prediction Display Size

**Default:** Shows prediction + full chart

**To change:** Edit `src/widgets.py` in `PredictionDisplay` class

```python
# Modify chart parameters
fig_width = 6
fig_height = 4
```

### Model Configuration

#### Number of Models to Load

**Default:** All .keras files in `models/` directory

**To limit:** Manually delete models from `models/` directory

**Or programmatically:** Edit `src/model_ensemble.py`

```python
# In ModelEnsemble.__init__()
max_models = 3  # Limit to 3 models
for i, model_file in enumerate(model_files):
    if i >= max_models:
        break
    # Load model...
```

**Considerations:**
- More models: Better accuracy but slower predictions
- 1 model: Fast but less accurate
- 3 models: Good balance (recommended)
- 5+ models: Diminishing returns on accuracy

#### Dataset for Training

**Default:** MNIST

**To use EMNIST:** Edit training call

```python
# In gui.py, train_new_model() method
ensemble.create_new_model(dataset_type='emnist')
```

**Options:**
- `'mnist'` - Standard MNIST (70k samples)
- `'emnist'` - Extended MNIST (800k+ samples)

**When to use:**
- MNIST: Faster training, good accuracy (default)
- EMNIST: Better generalization, slower training

#### Model Save Location

**Default:** `./models/`

**To change:** Edit `src/model_ensemble.py`

```python
# In ModelEnsemble.__init__()
self.models_dir = os.path.join(
    os.path.dirname(src_dir),
    'my_models'  # New directory name
)
```

### Training Configuration

#### Number of Epochs

**Default:** Up to 50 (with early stopping)

**To change:** Edit `src/model_ensemble.py`

```python
# In ModelEnsemble.create_new_model()
model.fit(
    x_train, y_train,
    epochs=25,  # Change this
    validation_split=0.2,
    callbacks=callbacks
)
```

**Considerations:**
- Too few (1-5): Underfitting
- Good range (10-50): Depends on data
- Too many (100+): Risk of overfitting

#### Batch Size

**Default:** 32 (Keras default)

**To change:** Edit training call

```python
model.fit(
    x_train, y_train,
    batch_size=64,  # Change this
    epochs=epochs,
    ...
)
```

**Effect:**
- Smaller batches (16): Noisier gradients, more exploration
- Larger batches (128): Smoother gradients, faster processing

#### Early Stopping Parameters

**Default:** Stop after 5 epochs without improvement

**To change:** Edit `src/model_ensemble.py`

```python
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,  # Change this
        restore_best_weights=True
    ),
    ...
]
```

#### Learning Rate Reduction

**Default:** Reduce by 80% after 3 plateauing epochs

**To change:**

```python
callbacks = [
    ...,
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,    # Multiply learning rate by 0.2
        patience=3,    # Epochs before reducing
        min_lr=0.0001  # Minimum learning rate
    )
]
```

### Preprocessing Configuration

#### Binary Threshold

**Default:** 200 (out of 255)

**To change:** Edit `src/digit_preprocessing.py`

```python
def preprocess_digit_image(img, preview_size=(140, 140)):
    _, binary = cv2.threshold(img, 200, cv2.THRESH_BINARY_INV)
    #                                   ^^^ Change this value
```

**Effect:**
- Lower (100): More background kept as "foreground"
- Higher (150): More sensitive, might miss light strokes
- 200: Good balance for typical handwriting

#### Digit Padding

**Default:** 25% of digit size

**To change:** Edit `src/digit_preprocessing.py`

```python
padding = max(w, h) // 4  # 1/4 = 25%
# Change divisor: // 2 = 50%, // 8 = 12.5%
```

#### Preview Image Size

**Default:** 140×140 pixels

**To change:** Edit `src/digit_preprocessing.py`

```python
def preprocess_digit_image(img, preview_size=(140, 140)):
    #                                             ^^^^^^^^^^^
    # Change to (200, 200), (100, 100), etc.
```

---

## Language & Localization

### Supported Languages

**Default:** English and Danish

**Available:**
- `'en'` - English
- `'da'` - Danish

### Adding New Language

1. Edit `src/gui.py`

```python
# Find language strings dictionary
LANGUAGES = {
    'en': {...},
    'da': {...},
    'es': {  # Add new language
        'title': 'Reconocedor de Dígitos CNN',
        'draw': 'Dibujar',
        # ... add all strings
    }
}
```

2. Add flag emoji

```python
LANGUAGE_FLAGS = {
    'en': '🇬🇧',
    'da': '🇩🇰',
    'es': '🇪🇸'  # Add flag
}
```

3. Update language toggle button

```python
def setup_language_buttons(self):
    for lang in ['en', 'da', 'es']:  # Add 'es'
        # ... create button
```

---

## GUI Customization

### Colors & Theme

**Default:** Neutral gray/white theme

**To customize:** Edit constants in `src/gui.py`

```python
# Add after imports
COLORS = {
    'bg_main': '#ffffff',          # Main background
    'bg_panel': '#f5f5f5',         # Panel background
    'fg_text': '#333333',          # Text color
    'highlight': '#4CAF50',        # Green highlight
    'button_hover': '#e0e0e0',     # Hover color
    'error': '#f44336'             # Error red
}
```

### Fonts

**Default:** System default fonts

**To customize:** Edit font definitions

```python
# In DigitRecognitionApp.__init__()
FONT_TITLE = ('Arial', 24, 'bold')
FONT_LARGE = ('Arial', 18, 'bold')
FONT_NORMAL = ('Arial', 10)
FONT_SMALL = ('Arial', 8)
```

### Button Styling

**To customize:** Edit widget creation in `src/widgets.py`

```python
button = tk.Button(
    parent,
    text='My Button',
    bg=COLORS['bg_panel'],
    fg=COLORS['fg_text'],
    font=FONT_NORMAL,
    padx=10,
    pady=5,
    relief=tk.FLAT,
    bd=1
)
```

---

## Performance Tuning

### Optimize for Speed

1. **Reduce model count:**
   - Remove extra models from `models/`
   - Keep only 1-2 models

2. **Reduce canvas size:**
   ```python
   canvas_width = 200
   canvas_height = 200
   ```

3. **Enable GPU (if available):**
   ```bash
   pip uninstall tensorflow
   pip install tensorflow[and-cuda]
   ```

4. **Close other applications:**
   - Free up system memory
   - Reduce CPU contention

### Optimize for Accuracy

1. **Add more models:**
   - Train multiple models
   - Larger ensemble improves accuracy

2. **Use EMNIST for training:**
   ```python
   ensemble.create_new_model(dataset_type='emnist')
   ```

3. **Collect diverse training data:**
   - Different handwriting styles
   - Different digit sizes/positions
   - 20-50 samples per digit

4. **Increase training epochs:**
   ```python
   epochs=50  # Instead of default
   ```

---

## Environment Variables

### Optional Configuration via Environment

**Model directory:**
```bash
set CNN_MODELS_DIR=C:\my\models  # Windows
export CNN_MODELS_DIR=/path/to/models  # Linux/macOS
```

**Then in code:**
```python
models_dir = os.getenv('CNN_MODELS_DIR', './models')
```

**Verbose logging:**
```bash
set CNN_DEBUG=1  # Windows
export CNN_DEBUG=1  # Linux/macOS
```

---

## Advanced Configuration

### Custom Data Augmentation

**Location:** `src/model_ensemble.py`

```python
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    shift_range=0.1,
    # Add more augmentations:
    shear_range=0.1,
    fill_mode='nearest',
    horizontal_flip=False  # Don't flip digits!
)
```

### Custom Loss Function

**Location:** `src/model.py`

```python
# Instead of:
model.compile(
    loss='sparse_categorical_crossentropy',
    ...
)

# Use:
model.compile(
    loss='categorical_crossentropy',  # Different loss
    ...
)
```

### Custom Optimizer

**Location:** `src/model.py`

```python
from tensorflow.keras.optimizers import SGD, RMSprop

# Instead of:
optimizer='adam'

# Use:
optimizer=SGD(learning_rate=0.01)
# or
optimizer=RMSprop(learning_rate=0.001)
```

---

## Configuration Checklist

Before deployment, verify:

- [ ] Canvas size appropriate for target hardware
- [ ] Number of models reasonable (1-5)
- [ ] Training parameters suitable for data
- [ ] Color scheme consistent
- [ ] Language strings complete
- [ ] Performance acceptable
- [ ] All tests pass

---

## Resetting to Defaults

To reset all customizations:

```bash
# Restore original files from git
git checkout src/gui.py src/widgets.py src/model.py

# Recreate models
rm -rf models/*
python create_models.py

# Clear cache
rm -rf __pycache__ src/__pycache__
```

---

See also: [Getting Started](Getting-Started.md), [Development Setup](Development-Setup.md), [User Guide](User-Guide.md)
