# User Guide

Complete guide to using the CNN Digit Recognizer application for drawing and predictions.

## Overview

The CNN Digit Recognizer is an interactive desktop application that uses artificial intelligence to recognize handwritten digits (0-9) that you draw on a canvas.

## Getting Started

### Launching the Application

```bash
python run_gui.py
```

The application window will appear with:
- **Left Panel:** White drawing canvas
- **Right Panel:** Prediction results and confidence chart
- **Bottom Panels:** Preview image, training controls, action buttons

## Main Features

### 1. Drawing Digits

#### Using Your Mouse

1. Click and drag in the **white canvas area**
2. Black lines appear as you draw
3. Draw a single digit (0-9)
4. Try to fill 60-80% of the canvas

#### Using a Stylus/Pen Tablet

For better control, use a drawing tablet instead of mouse:
- More precise lines
- Better resembles actual handwriting
- Improved accuracy from the AI

#### Tips for Best Results

✓ Draw clearly with continuous strokes  
✓ Center the digit roughly  
✓ Make digit reasonably large (not too small)  
✓ Use a steady hand (slow, deliberate strokes)  
✗ Don't make digit too thick  
✗ Don't cross lines over themselves  
✗ Don't use very unusual writing styles  

### 2. Making Predictions

#### Method 1: Keyboard Shortcut (Fastest)
```
Press Enter
    or
Press Ctrl+X
```

#### Method 2: Click Button
```
Click "Guess my digit!" button
```

The application will:
1. Extract your drawing from the canvas
2. Preprocess it (convert to MNIST format)
3. Run through the AI models
4. Display the predicted digit with confidence

#### Result Display

```
┌─────────────────────┐
│ Predicted: 5        │  ← The digit AI thinks it is
│ Confidence: 92.3%   │  ← Certainty level
├─────────────────────┤
│  Bar Chart (0-9)    │  ← All digit probabilities
│  0: ░░░░░░  2.1%    │
│  1: ░░░░░░  1.9%    │
│  ...                │
│  5: ████████ 92.3%  │  ← Green (highest)
│  ...                │
└─────────────────────┘
```

### 3. Clearing the Canvas

#### Method 1: Keyboard Shortcut
```
Press Ctrl+Z
```

#### Method 2: Click Button
```
Click "Clear Drawing" button
```

The canvas becomes white again, ready for a new digit.

### 4. Contributing Training Data

If the AI prediction is wrong, you can help it learn!

#### Step 1: Check/Set Correct Digit

After making a prediction, the prediction panel shows buttons for digits 0-9.

**If prediction is correct:**
- No action needed, skip to Step 3

**If prediction is wrong:**
- Click the button showing the correct digit
- Example: If AI said "5" but you drew "3", click button "3"

#### Step 2: Add Your Drawing to Training Data

Option A - Keyboard:
```
Press Ctrl+V
```

Option B - Click Button:
```
Click "Add my drawing to help the AI learn"
```

**What happens:**
- Your drawing is saved with the correct label
- Application displays: "Sample added to training data"
- Counter shows number of samples collected
- Drawing remains on canvas until you clear it

#### Step 3: Repeat to Collect More Data

```
Draw digit → Predict → Correct if wrong → Add data
         ↓
        Train when ready
```

### 5. Training New Models

Once you've collected several training samples (5-10+), teach the AI:

#### Method 1: Keyboard Shortcut
```
Press Ctrl+N
```

#### Method 2: Click Button
```
Click "Teach the AI with my drawings"
```

**What happens:**
1. Application processes all collected samples
2. Creates new CNN model
3. Trains model on your data + MNIST dataset
4. Saves model automatically
5. Model immediately added to ensemble
6. Next predictions use the new model
7. Training dialog shows progress

**Note:** Training takes 2-5 minutes depending on:
- Number of samples collected
- Your computer's speed
- GPU availability

**Status display:**
- ✓ Shows epoch number (1/50, 2/50, etc.)
- ✓ Shows training progress bar
- ✓ Shows validation accuracy
- ✓ "Complete!" when finished

### 6. Changing Language

#### Method 1: Click Flag Icon
- 🇬🇧 for English
- 🇩🇰 for Danish

#### Method 2: Keyboard
```
Look for flag icons in top-right of window
```

**What changes:**
- Button labels
- Help text
- Error messages
- Training data collection text

---

## Keyboard Shortcuts Reference

Quick reference for all keyboard shortcuts:

| Key(s) | Action | Notes |
|--------|--------|-------|
| `Ctrl+Z` | Clear Canvas | Start fresh |
| `Enter` | Make Prediction | Main action |
| `Ctrl+X` | Make Prediction | Alternative |
| `Ctrl+V` | Add to Training Data | Saves current drawing |
| `Ctrl+N` | Train New Model | Start training |
| `Click Flag` | Toggle Language | English ↔ Danish |

---

## Workflow Examples

### Scenario 1: Quick Prediction

```
1. Draw a digit
2. Press Enter
3. See prediction on right side
4. Enjoy! (Done)
```

### Scenario 2: Improve AI Accuracy

```
1. Draw digit "3"
2. Press Enter → AI predicts "5" (wrong!)
3. Click button "3" (correct digit)
4. Press Ctrl+V (add to training)
5. Repeat steps 1-4 ten more times with different digits
6. Press Ctrl+N (train new model)
7. Wait 2-5 minutes
8. Now AI is smarter!
```

### Scenario 3: Rapid Predictions

```
1. Draw digit
2. Press Enter → See result
3. Press Ctrl+Z → Clear
4. GOTO Step 1
```

---

## Understanding Results

### Prediction Display

#### High Confidence (>90%)
✓ Prediction likely correct  
✓ One bar clearly dominates  
✓ Other digits show minimal probability  

```
5: ███████████████ 92.3%
```

#### Medium Confidence (70-90%)
⚠ Prediction probably correct  
⚠ Reasonable confidence but some uncertainty  
⚠ Check if digit looks like predicted digit  

```
3: ██████████ 78.5%
```

#### Low Confidence (<70%)
⚠ Prediction uncertain  
⚠ Check digit quality  
⚠ Consider redrawing  

```
7: ██████ 64.2%
```

#### Ambiguous (Multiple high bars)
⚠ Could be multiple digits  
⚠ Digit is unclear  
⚠ Consider redrawing more clearly  

```
3: █████████ 45%
7: █████████ 44%
1: █████ 11%
```

### Why Predictions Sometimes Fail

| Reason | Example | Fix |
|--------|---------|-----|
| Unclear drawing | Squiggly mess | Redraw more carefully |
| Upside down | 9 looks like 6 | Orient digit correctly |
| Poor aspect ratio | Squashed digit | Use more canvas |
| Off-center | Digit way to the left | Center in canvas |
| Wrong handwriting | Unusual style | Practice standard writing |
| Empty canvas | Blank white area | Actually draw a digit |

---

## Training Data Tips

### Collecting High-Quality Data

**Do:**
- ✓ Use clear, recognizable writing
- ✓ Vary handwriting style (if it's yours)
- ✓ Collect digits from 0-9
- ✓ Make digits different sizes/positions
- ✓ Use 20-50 samples per digit for best results

**Don't:**
- ✗ Use completely illegible writing
- ✗ Collect only one digit (train balanced data)
- ✗ Use all identical drawings
- ✗ Collect fewer than 5 samples
- ✗ Use rotated/upside-down digits

### Training Progress

As you add more training data:
- Accuracy improves
- Model becomes personalized to YOUR handwriting
- May decrease accuracy for others' handwriting (unless diverse)
- Takes longer to train with more data

---

## Advanced Tips

### Tablet vs. Mouse

**Mouse:**
- Okay for testing
- Less precise
- Harder to control thickness
- Result: Lower accuracy

**Drawing Tablet:**
- Better line quality
- More natural handwriting
- Consistent pressure
- Result: Higher accuracy

### Stylus Pressure

Some tablets support pressure sensitivity:
- Lighter pressure = thinner lines
- Harder pressure = thicker lines
- Experiment to find optimal thickness

### Canvas Size

The canvas is **280×280 pixels**.

**Optimal digit size:**
- Height: 140-200 pixels (~50-70% of canvas)
- Width: 100-180 pixels
- Position: Roughly centered

---

## Troubleshooting

### I Can't Draw

**Problem:** Drawing doesn't appear on canvas

**Solutions:**
- Click in the white canvas area first
- Ensure drawing tool is selected
- Try a simple line first
- Check mouse/stylus is working

### Predictions Are Wrong

**Problem:** AI always predicts the wrong digit

**Solutions:**
1. Draw digit more clearly
2. Center digit better
3. Collect training data for improvement
4. Train new model with your data
5. Use larger digit size

### Predictions Are Slow

**Problem:** Predictions take 2+ seconds

**Solutions:**
- First prediction: Normal (models loading)
- Subsequent: Should be faster
- Close other applications
- Check system RAM usage

### Training Doesn't Start

**Problem:** Clicking train button does nothing

**Solutions:**
- Collect at least 5 training samples first
- Ensure samples are added (check counter)
- Try keyboard shortcut (Ctrl+N)
- Wait 10 seconds (might be processing)

### Language Won't Change

**Problem:** Clicking flag buttons doesn't change language

**Solutions:**
- Try clicking directly on flag emoji
- Check window is in focus
- Try all buttons, might need multiple clicks

---

## Tips for Best Accuracy

1. **Clear Drawing** - Legible, not messy
2. **Good Spacing** - No overlapping lines
3. **Proper Size** - Not too small, not huge
4. **Centered Position** - Roughly in middle
5. **Consistent Style** - Similar handwriting each time
6. **Use Tablet** - Better control than mouse
7. **Train AI** - Provide feedback for improvement
8. **Varied Data** - Different styles when training

---

## FAQ

**Q: Is my data stored?**  
A: Yes, locally on your computer only. Never sent to servers.

**Q: Can I delete training data?**  
A: Clear/rebuild the models directory to reset.

**Q: How accurate is this?**  
A: 95-98% on MNIST, 92-96% on user drawings.

**Q: What digits are supported?**  
A: Digits 0-9 only. Not letters.

**Q: Can I use this for other purposes?**  
A: Modify the code! It's open source.

**Q: How do I contribute to the project?**  
A: See GitHub repository for contribution guidelines.

---

See also: [Getting Started](Getting-Started.md), [Architecture Overview](Architecture.md), [Troubleshooting](Troubleshooting.md)
