# Architecture Overview

This document describes the high-level architecture and design patterns of the CNN Digit Recognizer application.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GUI Layer (Tkinter)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ Main Window │  │   Drawing    │  │ Prediction      │    │
│  │ (gui.py)    │  │   Canvas     │  │ Display/Chart   │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│             Application Logic Layer                        │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ DigitRecognitionApp Class                            │ │
│  │ - Event handling                                     │ │
│  │ - Threading for model predictions                   │ │
│  │ - State management                                  │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│          Model & Prediction Layer                          │
│  ┌──────────────────┐      ┌──────────────────────────┐   │
│  │ ModelEnsemble    │      │ Preprocessing Pipeline   │   │
│  │ - Load models    │      │ - Image conversion       │   │
│  │ - Averaging      │      │ - Binary thresholding    │   │
│  │ - Train new      │      │ - Digit extraction       │   │
│  │   models         │      │ - Normalization          │   │
│  └──────────────────┘      │ - MNIST formatting       │   │
│                             └──────────────────────────┘   │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│          Data & Persistence Layer                          │
│  ┌──────────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │ MNIST/EMNIST │  │  Model   │  │ Trained Models     │   │
│  │  Datasets    │  │ Files    │  │ (.keras format)    │   │
│  └──────────────┘  └──────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

### Prediction Pipeline

```
User Draws Digit
    ↓
Canvas.get_image() → Raw Pixel Array
    ↓
digit_preprocessing.preprocess_digit_image()
    ├─ Binary thresholding
    ├─ Contour detection
    ├─ Digit extraction
    ├─ Square padding
    ├─ Resize to 28x28
    └─ Normalize to [0,1]
    ↓
ModelEnsemble.predict()
    ├─ Load all models
    ├─ Get predictions from each model
    ├─ Average results
    └─ Return confidence scores
    ↓
GUI Update
    ├─ Display prediction
    ├─ Update confidence chart
    └─ Show preview image
```

### Training Pipeline

```
User Adds Training Data
    ↓
Application stores: (preprocessed_image, label)
    ↓
User Clicks "Teach the AI"
    ↓
ModelEnsemble.create_new_model()
    ├─ Combine existing training data
    ├─ Build new CNN architecture
    ├─ Train with callbacks:
    │  ├─ EarlyStopping
    │  └─ ReduceLROnPlateau
    └─ Save to models/ directory
    ↓
Model automatically added to ensemble
    ↓
Next predictions use updated ensemble
```

## Key Design Patterns

### 1. **Ensemble Pattern**
Multiple CNN models are loaded and their predictions are averaged for improved accuracy and robustness.

```python
# Instead of relying on single model prediction
single_pred = model.predict(image)

# Use ensemble averaging
predictions = [model.predict(image) for model in models]
ensemble_pred = np.mean(predictions, axis=0)
```

### 2. **Threading for UI Responsiveness**
Long-running operations (predictions, training) are executed in background threads to keep the UI responsive.

```python
thread = threading.Thread(target=self.predict_digit)
thread.daemon = True
thread.start()
```

### 3. **Preprocessing Pipeline**
A dedicated preprocessing module handles all image transformations to ensure consistency.

```
Raw Image → Binary → Contours → Extract → Pad → Normalize → Model Input
```

### 4. **Model Persistence**
Models are saved in Keras format (.keras) and automatically discovered from the `models/` directory.

```
ModelEnsemble.__init__() scans models/ directory → Loads all .keras files
```

## Module Dependencies

```
gui.py
├── model_ensemble.py
│   ├── model.py
│   ├── data_utils.py
│   └── tensorflow
├── widgets.py
│   ├── digit_preprocessing.py
│   └── matplotlib
└── digit_preprocessing.py
    ├── cv2
    └── PIL

model_ensemble.py
├── model.py
└── data_utils.py

widgets.py
└── digit_preprocessing.py
```

## Data Flow Example: Making a Prediction

1. **User Action**: User draws a digit and presses Enter
2. **Canvas Capture**: Drawing canvas image is extracted as numpy array
3. **Preprocessing**: Raw image is preprocessed to 28x28 MNIST format
4. **Threading**: Prediction task is submitted to background thread
5. **Model Ensemble**: Each model in ensemble makes a prediction
6. **Averaging**: Predictions are averaged (e.g., [0.1, 0.05, 0.8, ...])
7. **Display**: Results displayed in GUI with confidence chart
8. **Feedback**: User can indicate if prediction was correct

## Error Handling & Fallbacks

- **Model Loading**: If a model fails to load, it's skipped and application continues with remaining models
- **EMNIST Dataset**: Falls back to MNIST if EMNIST loading fails
- **Image Preprocessing**: Gracefully handles empty canvases and extreme digit sizes

## Performance Considerations

- **Model Caching**: Models are loaded once at startup, not on each prediction
- **Batch Operations**: Predictions use batch dimension for efficiency
- **Threading**: UI remains responsive during long operations
- **Image Resizing**: Images are normalized to fixed 28x28 size for consistent performance

## Extensibility Points

1. **Add New Models**: Drop .keras files in `models/` directory
2. **Custom Preprocessing**: Extend `digit_preprocessing.py` functions
3. **New Datasets**: Modify `data_utils.py` to support different datasets
4. **UI Themes**: Customize widget styling in `widgets.py`
5. **Additional Languages**: Add translations in `gui.py`

---

See also: [Core Modules](Core-Modules.md), [GUI Components](GUI-Components.md), [Model System](Model-System.md)
