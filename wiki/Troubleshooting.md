# Troubleshooting Guide

Solutions for common issues encountered with the CNN Digit Recognizer.

## Installation Issues

### ModuleNotFoundError: No module named 'tensorflow'

**Problem:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Causes:**
- Requirements not installed
- Virtual environment not activated
- Incorrect Python version

**Solutions:**

**1. Reinstall all dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

**2. Verify virtual environment is activated:**

Windows:
```powershell
# Should show (venv) or similar before prompt
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

**3. Check Python version:**
```bash
python --version
# Should be 3.7 or higher
```

**4. Try installing specific packages:**
```bash
pip install tensorflow opencv-python pillow matplotlib
```

### Tkinter Not Found

**Problem:**
```
ImportError: No module named 'tkinter'
```

**Causes:**
- Tkinter not installed with Python
- Python installation incomplete
- Missing system package

**Solutions:**

**Windows:**
```powershell
python -m pip install --upgrade tk
```

**macOS:**
```bash
# Usually included, if missing reinstall Python from python.org
brew install python-tk@3.11
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3-tk
```

**Linux (Fedora/RHEL):**
```bash
sudo yum install python3-tkinter
```

### Pillow/PIL Import Error

**Problem:**
```
ImportError: cannot import name 'ImageDraw' from 'PIL'
```

**Solution:**
```bash
pip install --upgrade Pillow
```

---

## Application Launch Issues

### No GUI Window Appears

**Problem:** Application runs but no window shows

**Possible Causes:**
1. Display server not available (Linux)
2. Window rendered off-screen
3. GUI crashed silently

**Solutions:**

**1. Check for error output:**
```bash
python run_gui_debug.py 2>&1
# Look for error messages
```

**2. Try explicit display (Linux only):**
```bash
DISPLAY=:0 python run_gui.py
```

**3. Force window to visible (Windows):**
```bash
python -c "from src.gui import DigitRecognitionApp; app = DigitRecognitionApp(); app.geometry('+100+100'); app.mainloop()"
```

**4. Check system resources:**
```bash
# Windows
wmic OS get TotalVisibleMemorySize, FreePhysicalMemory

# Linux
free -h

# macOS
vm_stat
```

### Application Crashes on Startup

**Problem:** Application starts but crashes immediately

**Symptoms:**
- Window appears briefly then disappears
- Terminal shows stack trace
- Application exits with error code

**Solutions:**

**1. Run debug version:**
```bash
python run_gui_debug.py
```

**2. Check for missing models:**
```bash
python verify_setup.py
```

**3. Verify installation:**
```bash
python diagnose.py
```

**4. Check for path issues:**
```bash
python debug_paths.py
```

**5. Run in verbose mode:**
```bash
python -u run_gui.py 2>&1 | tee debug.log
```

### Memory Error on Startup

**Problem:**
```
MemoryError
OSError: [Errno 12] Out of memory
```

**Causes:**
- Insufficient RAM
- Too many models trying to load
- Memory leak in system

**Solutions:**

**1. Check available memory:**
```bash
python -c "import psutil; print(psutil.virtual_memory())"
```

**2. Reduce number of models:**
```bash
# Remove large models from models/ directory
# Keep only 1-2 models
```

**3. Close other applications**
- Chrome, VS Code, etc.
- Free up 2+ GB RAM

**4. Create swap space (Linux only):**
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Model Issues

### No Models Found Warning

**Problem:**
```
[ModelEnsemble] Found .keras files: []
No models loaded!
```

**Causes:**
- models/ directory empty
- Models not yet created
- Wrong directory path

**Solutions:**

**1. Create models:**
```bash
python create_models.py
# Wait 10-30 minutes
```

**2. Download pre-trained models:**
```bash
# Check GitHub releases for downloadable models
# Download and place in models/ directory
```

**3. Verify path:**
```bash
python debug_paths.py
# Check if models/ path is correct
```

**4. Check file extension:**
```bash
# Ensure model files end with .keras
# Not .h5 or .pb
```

### Model Loading Failed

**Problem:**
```
Error loading model_mnist_0.keras: [error details]
```

**Possible Causes:**
- File corrupted
- TensorFlow version mismatch
- File format issue

**Solutions:**

**1. Verify model file:**
```python
import tensorflow as tf
try:
    model = tf.keras.models.load_model('models/model_mnist_0.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: {e}")
```

**2. Retrain model:**
```bash
# Delete corrupted model
rm models/model_mnist_0.keras

# Retrain
python create_models.py
```

**3. Check TensorFlow version:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Should be 2.13+
# Update if needed
pip install --upgrade tensorflow
```

---

## Prediction Issues

### Predictions Always Wrong

**Problem:** AI consistently predicts wrong digits

**Causes:**
1. Handwriting style differs from training data
2. Digit not drawn clearly
3. Preprocessing issue
4. Model not trained

**Solutions:**

**1. Draw more clearly:**
- Clear, continuous strokes
- No erasing/rewriting
- Consistent line thickness

**2. Center digit better:**
- Position in middle of canvas
- Not too close to edges
- Roughly same size each time

**3. Check preprocessing:**
```python
from src.digit_preprocessing import preprocess_digit_image
import numpy as np

# Get drawing
raw = canvas.get_image()

# Check preprocessing
processed, preview = preprocess_digit_image(raw)

# Verify shape and values
print(f"Shape: {processed.shape}")  # Should be (1, 28, 28, 1)
print(f"Min/Max: {processed.min()}/{processed.max()}")  # Should be [0, 1]
```

**4. Verify models loaded:**
```bash
python test_model_load.py
```

**5. Train with your handwriting:**
- Collect 20-30 samples of each digit
- Train new model
- Accuracy should improve significantly

### Predictions Are Very Slow

**Problem:** Predictions take 3+ seconds

**Causes:**
1. Models still loading
2. GPU not available
3. System CPU limited
4. Too many models loaded

**Solutions:**

**1. First prediction takes longer**
- Model initialization: 1-2 seconds
- Subsequent predictions: 0.3-1 second
- Normal behavior

**2. Reduce model count:**
```bash
# Keep only 1-2 models
# Delete extras from models/ directory
```

**3. Close background applications:**
- Chrome, VS Code, etc.
- Task Manager → End task
- Free up CPU

**4. Enable GPU (optional):**
```bash
# If NVIDIA GPU available
pip install tensorflow[and-cuda]
```

**5. Use CPU affinity (Linux/macOS):**
```bash
# Run on specific cores
taskset -c 0-3 python run_gui.py
```

### Inconsistent Predictions

**Problem:** Same digit produces different predictions

**Causes:**
1. Drawing style varies too much
2. Digit placement inconsistent
3. Model ensemble averaging normal variation

**Solutions:**

**1. Draw more consistently:**
- Same size each time
- Same position
- Same handwriting style

**2. This is normal:**
- Ensemble averaging produces variation
- Confidence should still be high for correct digit

**3. Collect more training data:**
- 20-30 samples per digit
- Variety in position/size
- Train new model

---

## Training Issues

### Training Button Does Nothing

**Problem:** Click train but nothing happens

**Causes:**
1. Not enough training samples
2. Training already running
3. No samples actually added

**Solutions:**

**1. Check sample count:**
- Status should show number > 0
- If 0, add samples first

**2. Add training data:**
```
Draw → Predict → Select correct digit → Ctrl+V
```

**3. Verify samples added:**
- Check status display
- Counter should increase

**4. Try different trigger method:**
- Try Ctrl+N if button doesn't work
- Check window focus

### Training Crashes or Hangs

**Problem:** Training starts but crashes or freezes

**Causes:**
1. Insufficient memory
2. Data format issue
3. Model building error

**Solutions:**

**1. Monitor memory during training:**
```bash
# On another terminal
watch -n 1 free -h
```

**2. Collect fewer samples for first training:**
- Start with 5-10 samples
- Increase gradually

**3. Check training output:**
```bash
python run_gui_debug.py
# Monitor console during training
```

**4. Increase swap space (if needed):**
```bash
# Linux only
sudo fallocate -l 4G /swapfile
```

### Training Takes Too Long

**Problem:** Training runs for 20+ minutes

**Typical Times:**
- 5-10 samples: 2-3 minutes
- 20-30 samples: 5-8 minutes
- 50+ samples: 10-15 minutes

**To Speed Up:**

**1. Start with fewer samples:**
- 5 per digit instead of 20
- Increase gradually

**2. Close other applications:**
- Frees CPU resources

**3. Reduce epoch count:**
```python
# In model_ensemble.py
model.fit(..., epochs=10)  # Default is 50
```

**4. Enable GPU (if available):**
```bash
pip install tensorflow[and-cuda]
```

---

## GUI Issues

### Canvas Won't Draw

**Problem:** Can't draw on canvas

**Causes:**
1. Click outside canvas area
2. Canvas not focused
3. Input device issue

**Solutions:**

**1. Click directly on white area:**
- Ensure clicking inside white rectangle

**2. Clear and retry:**
```
Ctrl+Z to clear
Try again
```

**3. Check mouse/stylus:**
- Test mouse elsewhere
- Try different input device

**4. Restart application:**
```bash
python run_gui.py
```

### Language Toggle Not Working

**Problem:** Flag buttons don't change language

**Causes:**
1. Button not clickable
2. No visible change
3. Window focus lost

**Solutions:**

**1. Click directly on flag emoji:**
- 🇬🇧 for English
- 🇩🇰 for Danish

**2. Try multiple times:**
- May need 2-3 clicks

**3. Try keyboard fallback:**
- Check if keyboard shortcut available
- Or implement shortcut if needed

**4. Restart application:**
```bash
python run_gui.py
```

### Chart Display Issues

**Problem:** Confidence chart not showing or looks wrong

**Causes:**
1. Matplotlib rendering issue
2. Display server problem
3. Tkinter canvas issue

**Solutions:**

**1. Make prediction first:**
- Chart only appears after first prediction

**2. Check for errors:**
```bash
python run_gui_debug.py
```

**3. Try different backend (advanced):**
```python
# In src/widgets.py
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

---

## Data & File Issues

### Training Data Lost

**Problem:** Training samples disappeared

**Note:** Samples are stored in application memory during session

**Solutions:**

**1. Samples only during session:**
- Restart app = samples lost
- This is by design
- Retraining creates new models from scratch

**2. If models are lost:**
```bash
# Models saved in models/ directory
# Check directory exists and has files
dir models/
```

### Disk Space Issues

**Problem:** Application says disk full

**Typical Model Size:** 50-100 MB each

**Solutions:**

**1. Check disk space:**
```bash
# Windows
dir C:\

# Linux/macOS
df -h
```

**2. Delete old models:**
```bash
# Keep only latest models
rm models/model_mnist_*.keras  # Delete old MNIST
```

**3. Clean up system:**
- Delete temporary files
- Empty recycle bin
- Remove unused programs

---

## Performance Issues

### Application Laggy or Slow

**Problem:** UI feels slow/unresponsive

**Causes:**
1. System overloaded
2. Too many models
3. High resolution display

**Solutions:**

**1. Close other applications:**
- Chrome, teams, Discord
- Free up RAM

**2. Reduce model count:**
- Keep 1-2 models only

**3. Reduce canvas size (if possible):**
- Edit run_gui.py
- Change canvas dimensions

### High CPU Usage

**Problem:** Application using 100% CPU

**Causes:**
1. Model inference running
2. Training in progress
3. Infinite loop bug

**Solutions:**

**1. Wait for operation to complete:**
- Training: 2-15 minutes
- Prediction: 0.1-1 seconds

**2. Force close if hung:**
```bash
# Windows
taskkill /IM python.exe

# Linux/macOS
pkill -9 python
```

**3. Check for bugs:**
```bash
python run_gui_debug.py
# Look for exception in console
```

---

## Getting More Help

### Debugging Commands

```bash
# Full diagnostics
python diagnose.py

# Verify setup
python verify_setup.py

# Path debugging
python debug_paths.py

# Model loading test
python test_model_load.py

# Run tests
pytest tests/ -v

# Debug output
python run_gui_debug.py
```

### Where to Look

| Issue | Check |
|-------|-------|
| Installation | requirements.txt, Python version |
| Models | models/ directory, file names |
| GUI | src/gui.py, src/widgets.py |
| Prediction | src/model_ensemble.py, test_models.py |
| Preprocessing | src/digit_preprocessing.py |
| Training | src/model_ensemble.py, training.py |

### Report Issues

If problem persists:

1. **Collect information:**
   - Python version
   - Error messages
   - System info (OS, RAM, CPU)
   - Steps to reproduce

2. **Check existing issues:**
   - GitHub Issues page

3. **Create new issue:**
   - Include all above information
   - Provide full error traceback

---

See also: [User Guide](User-Guide.md), [Getting Started](Getting-Started.md), [Architecture Overview](Architecture.md)
