# GUI Components Guide

Comprehensive guide to the Tkinter UI components and layout of the CNN Digit Recognizer.

## GUI Overview

The application uses **Tkinter** for the user interface, organized in a main window with several custom widgets and panels.

```
┌─────────────────────────────────────────────────────┐
│           CNN DIGIT RECOGNIZER                      │
├──────────────────┬──────────────────────────────────┤
│                  │                                  │
│   Drawing Canvas │      Prediction Display         │
│   (280×280)      │      - Large Digit              │
│                  │      - Confidence Chart         │
│                  │      - Probability Bars         │
├──────────────────┴──────────────────────────────────┤
│          Preview Panel (140×140)                    │
│     Preprocessed MNIST Format (28×28)               │
├─────────────────────────────────────────────────────┤
│          Training Panel                             │
│  [Correct Digit: 0 1 2 3 4 5 6 7 8 9]              │
│  [Add to Training] [Train New Model]               │
├─────────────────────────────────────────────────────┤
│  [Guess my digit!] [Clear] [🇬🇧/🇩🇰]             │
└─────────────────────────────────────────────────────┘
```

## Custom Widget Classes

### 1. DrawingCanvas

**Location:** `src/widgets.py`  
**Inherits from:** `tk.Canvas`

Provides an interactive canvas for drawing digits with mouse or stylus.

#### Features
- **Dimensions:** 280×280 pixels by default
- **Line Width:** 12 pixels (customizable)
- **Color:** Black on white background
- **Smooth Rendering:** Uses tk.ROUND capstyle for smooth lines
- **Dual Storage:** Maintains both Tkinter canvas and PIL image

#### API

**Properties:**
```python
canvas.width            # Canvas width in pixels
canvas.height           # Canvas height in pixels
canvas.line_width       # Brush size (default 12)
canvas.image            # PIL Image object
canvas.draw             # PIL ImageDraw object
```

**Methods:**
```python
# Get drawing as numpy array
image_array = canvas.get_image()  # Returns shape (280, 280)

# Clear the canvas
canvas.clear()

# (Internal) Paint on canvas
canvas.paint(event)

# (Internal) Reset drawing state
canvas.reset_last_point(event)
```

#### Implementation Details

The DrawingCanvas maintains parallel representations:
1. **Tkinter Canvas** - Visual display on screen
2. **PIL Image** - Actual pixel data for processing

This dual storage allows for:
- Real-time visual feedback
- Efficient image processing
- Easy numpy conversion

#### Example Usage

```python
from src.widgets import DrawingCanvas

# Create canvas
canvas = DrawingCanvas(
    parent_frame,
    width=280,
    height=280,
    bg='white'
)

# Get drawing for processing
raw_image = canvas.get_image()  # numpy array (280, 280)

# Clear when done
canvas.clear()
```

#### Event Binding

Automatically binds:
- `<B1-Motion>` - Mouse drag (paint)
- `<ButtonRelease-1>` - Mouse release (finish stroke)

---

### 2. PredictionDisplay

**Location:** `src/widgets.py`  
**Inherits from:** `tk.Frame`

Displays model predictions with visualization.

#### Features
- **Large Prediction Text** - Shows predicted digit in large font
- **Confidence Score** - Displays as percentage
- **Bar Chart** - Matplotlib visualization of all 10 digit probabilities
- **Color Coding:** 
  - Green bar for predicted digit
  - Gray bars for other digits
- **Real-time Updates** - Updates when new predictions arrive

#### Structure

```
┌──────────────────────────┐
│  Predicted: 5            │  ← Large text display
│  Confidence: 94.2%       │  ← Percentage
├──────────────────────────┤
│  [Matplotlib Bar Chart]  │  ← Probabilities for 0-9
│  0: ░░░░░░░░  3.2%       │
│  1: ░░░░░░░░  2.1%       │
│  2: ░░░░░░░░  4.5%       │
│  3: ░░░░░░░░  1.8%       │
│  4: ░░░░░░░░  2.9%       │
│  5: ████████████ 94.2%   │  ← Green (highest)
│  6: ░░░░░░░░  1.3%       │
│  7: ░░░░░░░░  2.1%       │
│  8: ░░░░░░░░  3.4%       │
│  9: ░░░░░░░░  2.5%       │
└──────────────────────────┘
```

#### API

**Methods:**
```python
# Update display with new predictions
display.update_prediction(predictions, predicted_digit)
# predictions: numpy array shape (10,)
# predicted_digit: int 0-9

# Clear display
display.clear()
```

#### Implementation

Uses **Matplotlib** with Tkinter backend:
```python
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
```

The chart is embedded directly in the Tkinter window without creating separate windows.

---

### 3. TrainingPanel

**Location:** `src/widgets.py`  
**Inherits from:** `tk.Frame`

Interface for collecting and managing training data.

#### Features
- **Digit Selector** - Buttons for digits 0-9 to indicate correct answer
- **Add Button** - Submit drawing to training dataset
- **Train Button** - Trigger new model training
- **Status Display** - Show number of training samples collected
- **Language Support** - Multilingual button labels

#### Structure

```
┌──────────────────────────────────┐
│ Is this correct?                 │
│ [0] [1] [2] [3] [4] [5] [6]     │  ← Digit selection
│ [7] [8] [9]                      │
│                                  │
│ [Add my drawing to help the AI] │  ← Add training sample
│ [Teach the AI with my drawings] │  ← Train new model
│                                  │
│ Samples collected: 15            │  ← Status
└──────────────────────────────────┘
```

#### Signals/Events

**When user clicks a digit button:**
- Selected digit is stored
- User must click "Add drawing" to confirm

**When user clicks "Add drawing":**
- Current canvas image + selected digit added to training data
- Count updated

**When user clicks "Train":**
- Background thread spawned
- `create_new_model()` called with collected data
- New model automatically added to ensemble

#### Example Usage

```python
training_panel = TrainingPanel(parent_frame, app)
# Automatically connected to parent app's training_data list
```

---

## Main Application Layout

### DigitRecognitionApp Window Structure

The main window layout is organized with **ttk.Frame** containers:

```python
class DigitRecognitionApp(tk.Tk):
    def __init__(self):
        # Main sections:
        self.header_frame       # Logo, language buttons
        self.content_frame      # Canvas + prediction display
        self.preview_frame      # Preprocessed digit display
        self.training_frame     # Training controls
        self.button_frame       # Main action buttons
```

### Key Layout Sections

#### 1. Header Frame (Top)
- Application title
- Language toggle buttons (🇬🇧 / 🇩🇰)
- Tooltips for language options

#### 2. Content Frame (Middle-Large)
- **Left Side:** DrawingCanvas (280×280)
- **Right Side:** PredictionDisplay with chart

**Layout Code Pattern:**
```python
content_frame = ttk.Frame(root)
content_frame.pack(side=tk.LEFT, padx=10, pady=10)

canvas = DrawingCanvas(content_frame, width=280, height=280)
canvas.pack()

prediction_display = PredictionDisplay(content_frame)
prediction_display.pack()
```

#### 3. Preview Frame (Middle-Small)
- Shows 140×140 preview of preprocessed digit
- Displays how digit will look to the model (MNIST format)
- Label with text "Preprocessed Preview: (28×28 MNIST)"

#### 4. Training Frame (Lower-Middle)
- TrainingPanel widget
- Digit selection buttons
- Training data collection controls

#### 5. Button Frame (Bottom)
- Main action buttons:
  - "Guess my digit!" - Trigger prediction
  - "Clear Drawing" - Clear canvas
  - Status labels
- Keyboard shortcut reminders

---

## Widget Styling & Theme

### Colors Used
```python
CANVAS_BG = 'white'
BUTTON_BG = '#f0f0f0'
PREDICTION_BG = '#f5f5f5'
TEXT_COLOR = '#333333'
HIGHLIGHT_COLOR = '#4CAF50'  # Green
ERROR_COLOR = '#f44336'      # Red
```

### Fonts
```python
TITLE_FONT = ('Arial', 24, 'bold')
LARGE_FONT = ('Arial', 18, 'bold')
NORMAL_FONT = ('Arial', 10)
SMALL_FONT = ('Arial', 8)
```

### Button Styling
- Flat design with subtle borders
- Hover effects using `<Enter>` / `<Leave>` events
- Active state changes background
- Rounded corners via paddings

---

## Responsive Design Considerations

### Canvas Scaling
- Fixed 280×280 for consistency with MNIST preprocessing
- Scales with window but maintains aspect ratio
- Touch/tablet friendly with 12-pixel line width

### Chart Display
- Matplotlib figure embedded in Tkinter
- Automatically scales with frame size
- High DPI rendering for clarity

### Text Labels
- Dynamic sizing based on predicted confidence
- Color intensity varies with confidence level

---

## Threading & Updates

### Prediction Update Pattern

When user makes a prediction:
1. Background thread executes `predict_digit()`
2. Thread updates GUI via `after()` callback
3. Prevents UI freezing during model inference

```python
def predict_digit(self):
    # Background thread
    predictions = self.model_ensemble.predict(image)
    # Queue GUI update
    self.after(0, lambda: self.display_update(predictions))
```

### Why This Matters
- Model prediction can take 1-3 seconds
- Without threading, UI would freeze
- User would see "Not Responding" warning

---

## Accessibility Features

### Keyboard Shortcuts
See [User Guide - Keyboard Shortcuts](User-Guide.md#keyboard-shortcuts)

### Tooltips
- Hover over buttons for help text
- Implemented via custom ToolTip class
- Language-aware

### High Contrast
- Black/white contrast for drawing canvas
- Clear separation between UI sections

---

## Internationalization (i18n)

### Language Support
- English (en)
- Danish (da)

### Implementation
- Text strings stored in language dictionaries
- `toggle_language()` updates all labels
- Flag emoji buttons for selection
- File: `src/gui.py` contains translation strings

### Adding New Language
1. Add language code to `LANGUAGES` dict
2. Add flag emoji for language
3. Update all translatable strings in gui.py

---

See also: [Core Modules](Core-Modules.md), [User Guide](User-Guide.md), [Architecture Overview](Architecture.md)
