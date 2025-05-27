# Handwritten Digit Recognition GUI

An interactive desktop application for handwritten digit recognition using ensemble machine learning models. Users can draw digits, get AI predictions, and contribute training data to improve the model's accuracy.

![image](https://github.com/user-attachments/assets/e3e8e379-d921-4f99-8cd9-7bf72a835f5d)

## Features

- **Interactive Drawing Canvas**: Draw digits using mouse or drawing tablet
- **Real-time Preprocessing**: See how your drawing is processed for the AI (28x28px, MNIST format)
- **Ensemble Model Predictions**: Uses multiple CNN models for improved accuracy
- **Confidence Visualization**: Bar chart showing prediction probabilities for all digits (0-9)
- **Interactive Training**: Add your drawings to help improve the AI
- **Multi-language Support**: English and Danish* interface        |  *Still missing some translations
- **Model Management**: Automatic ensemble creation from multiple trained models*  | *This is still not working properly

## How It Works

### User Workflow
1. **Draw**: Use mouse/tablet to draw a single digit (0-9) in the white canvas
2. **Predict**: Click "Guess my digit!" to see the AI's prediction with confidence levels
3. **Teach**: If the AI guesses wrong, select the correct digit and add your drawing to training data
4. **Improve**: Click "Teach the AI with my drawings" to create a new model with your contributions

### Technical Process
1. **Preprocessing**: Raw drawings are converted to 28x28px grayscale images, centered and normalized to match MNIST format
2. **Prediction**: Ensemble of CNN models makes predictions, results are averaged for final confidence scores
3. **Training**: New models are trained on user-contributed data and automatically added to the ensemble
4. **Ensemble**: Multiple models work together using prediction averaging for improved accuracy

## Installation

### Prerequisites
- Python 3.7 or higher (we recommened 3.11)
- pip package manager

### Dependencies
Install required packages:

```bash
pip install tensorflow opencv-python pillow matplotlib scikit-learn numpy tkinter tensorflow-datasets
```

### Setup
1. Clone this repository:
```bash
git clone https://github.com/w1setown/CNN-Digit-Recognize
cd handwritten-digit-recognition
```

2. Create the models directory:
```bash
mkdir models
```

3. (Optional) Create initial models:
```bash
python create_models.py
```

## Usage

### Starting the Application
```bash
python gui.py
```

### Controls and Shortcuts
- **Drawing**: Click and drag in the white canvas area
- **Clear Canvas**: `Ctrl+Z` or click "Clear Drawing"
- **Make Prediction**: `Enter` or `Ctrl+X` or click "Guess my digit!"
- **Add Training Data**: `Ctrl+V` or click "Add my drawing to help the AI learn"
- **Train New Model**: `Ctrl+N` or click "Teach the AI with my drawings"
- **Language Toggle**: Click the flag icons (🇬🇧/🇩🇰) to switch between English and Danish

### Using a Drawing Tablet
For best results simulating handwriting, use a drawing tablet instead of a mouse.

## File Structure

```
CNN-DIGIT-RECOGNIZER/
├── __pycache__/                # Python cache files
├── models/                     # Saved model files
│   └── model_mnist_0.keras     # Trained Keras model
├── test_digits/                # Test digit images
├── create_models.py            # Script to create initial models
├── data_utils.py               # Data loading and preprocessing utilities
├── digit_preprocessing.py      # Image preprocessing functions
├── flag_dk.png                 # Danish flag image
├── flag_uk.png                 # UK flag image
├── gui.py                      # Main application window and UI logic
├── model_ensemble.py           # Ensemble model management and predictions
├── model_evaluation.py         # Model evaluation and metrics
├── model.py                    # CNN model architecture definition
├── requirements.txt            # Project dependencies
├── training.py                 # Model training script
├── widgets.py                  # Custom UI components (canvas, charts, panels)
└── README.md                   # This file
```

## Model Architecture

The application uses Convolutional Neural Networks (CNNs) with the following architecture:

```python
Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')  # 10 classes for digits 0-9
])
```

### Training Features
- **Data Augmentation**: Rotation, shifting, and zooming for robustness
- **Early Stopping**: Prevents overfitting during training
- **Learning Rate Reduction**: Adaptive learning rate scheduling
- **Ensemble Learning**: Multiple models combined for better accuracy

## Ensemble System

The application uses an ensemble approach where:
- Multiple CNN models are trained on different data (MNIST, user contributions)
- Predictions from all models are averaged for final results
- New models are automatically added to the ensemble when trained
- Model files are stored in the `models/` directory as `.keras` files

## Data Processing

### Input Processing
1. **Canvas Drawing**: Raw mouse/tablet input captured as PIL Image
2. **Binarization**: Converted to binary using thresholding (200 threshold)
3. **Contour Detection**: Find digit boundaries for cropping
4. **Centering**: Digit centered in bounding box with padding
5. **Resizing**: Scaled to 20x20px, then padded to 28x28px
6. **Normalization**: Pixel values normalized to 0-1 range

### MNIST Compatibility
All processing ensures compatibility with the MNIST dataset format:
- 28x28 pixel grayscale images
- White digits on black background
- Centered and size-normalized
- Float32 values in range [0, 1]

## Multilingual Support

The application supports:
- **English**: Default interface language
- **Danish**: Alternative interface with flag toggle

Language switching affects:
- All UI text and labels
- Button text and tooltips
- Status messages and dialogs
- Instructions and help text

## Contributing Training Data

### Data Collection
- Users draw digits and specify the correct label
- Drawings are preprocessed to MNIST format
- Data is stored temporarily until model training

### Training Process
1. User adds multiple labeled drawings
2. Click "Teach the AI with my drawings"
3. New CNN model is trained on user data
4. Model is automatically added to ensemble
5. Training data is cleared after successful training

### Best Practices
- Add multiple examples of each digit for better training
- Draw clearly and ensure digits are well-formed
- Use a drawing tablet for more natural handwriting
- Correct mislabeled predictions to improve accuracy

## Technical Details

### Dependencies
- **TensorFlow**: Deep learning framework and model training
- **OpenCV**: Image processing and preprocessing
- **Tkinter**: GUI framework (built into Python)
- **PIL/Pillow**: Image manipulation and display
- **Matplotlib**: Prediction confidence charts
- **NumPy**: Numerical computations
- **scikit-learn**: Data splitting utilities

### Performance
- Models are loaded once at startup
- Predictions are near-instantaneous
- Training new models takes 1-3 minutes depending on data size
- Memory usage scales with number of models in ensemble

## Troubleshooting

### Common Issues

**"Models are still loading"**
- Wait for the initial model loading to complete
- Check that TensorFlow is properly installed

**"Please draw a digit first!"**
- Ensure you've drawn something in the white canvas area
- Try drawing with thicker strokes

**Training fails**
- Ensure you have sufficient training examples
- Check available disk space for model files
- Verify TensorFlow installation

**Poor prediction accuracy**
- Use a drawing tablet instead of mouse for better input
- Draw digits clearly and centered
- Add more training examples for problematic digits

### System Requirements
- **RAM**: Minimum 4GB (8GB recommended for training)
- **Storage**: At least 1GB free space for models
- **CPU**: Multi-core processor recommended for training
- **Input**: Mouse or drawing tablet

## License
MIT License

Copyright (c) 2025 Gabriel Visby Søgaard Ganderup & Jakob Lykke Lyngsøe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


## Acknowledgments

- Built with TensorFlow and Keras
- Uses MNIST and EMNIST datasets for initial training
- GUI built with Python Tkinter
- Image processing with OpenCV and PIL
- Shan Carter and Michael Nielsen for their work on augmenting human intelligence through AI.
- Adam Dhalla for his accessible and comprehensive teaching on the mathematics of back propagation in neural networks.
- Andrej Karpathy, Younes Bensouda, and Andrew Ng for their outstanding teaching through Coursera.
- Michael Nielsen, again, for his continuing contributions to both neural network education and philosophical reflections on AI and human progress.
