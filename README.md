# CNN Digit Recognizer

A real-time digit recognition server that uses a Convolutional Neural Network (CNN) to recognize handwritten digits from multiple clients.

## Features

- Real-time digit recognition via webcam
- Web interface for monitoring connected clients
- Training interface for adding new samples
- Model retraining capability
- TCP server for handling camera streams
- RESTful API endpoints for predictions

## Requirements

- Python 3.7 or higher
- GPU support (optional but recommended for faster training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CNN-Digit-Recognizer.git
cd CNN-Digit-Recognizer
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a directory for training data uploads:
```bash
mkdir training_uploads
```

2. Update the server IP address in `server.py` if needed:
```python
app.run(host='192.168.10.10', port=8000, threaded=True)
tcp_server.bind(('192.168.10.10', tcp_port))
```

## Usage

1. Start the server:
```bash
python server.py
```

2. Access the web interface:
- Open a web browser and navigate to `http://192.168.10.10:8000`

3. Available endpoints:
- `/` - Main web interface
- `/video_feed/<client_id>` - Video stream for specific client
- `/predict` - REST API for digit prediction
- `/upload_training_data` - Upload new training samples
- `/retrain_model` - Retrain the model with new data
- `/status` - Server status information
- `/model` - Download the trained model

## Training Data

- Upload training images through the web interface
- Supported formats: PNG, JPG, JPEG
- Images are automatically preprocessed and normalized
- Training data is stored in `training_uploads/` directory

## Model

- Uses a CNN architecture for digit recognition
- Supports both sparse and regular categorical crossentropy
- Model file is saved as `cnn_model.keras`
- Automatic model initialization on server start