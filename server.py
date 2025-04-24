from flask import Flask, Response, jsonify, send_file, request, render_template_string
import socket
import threading
import numpy as np
import cv2
import os
import io
import time
import json
from PIL import Image
import tensorflow as tf
from model import load_or_train_model
from training import retrain_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global variables
model = None
tcp_server = None
tcp_port = 8001
client_frames = {}  # Store frames from connected clients
client_predictions = {}  # Store predictions for connected clients

# Configuration for file uploads
UPLOAD_FOLDER = 'training_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Training data storage
training_images = []
training_labels = []

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    """Initialize the CNN model"""
    global model
    model = load_or_train_model()
    return model is not None

def process_digit_image(img_data):
    """Process digit image and return prediction"""
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Normalize and reshape for CNN
        processed_img = img.astype('float32') / 255
        processed_img = np.expand_dims(processed_img, axis=-1)  # Add channel dimension
        processed_img = np.expand_dims(processed_img, axis=0)   # Add batch dimension
        
        # Make prediction
        prediction = model.predict(processed_img, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        return {
            'digit': int(predicted_digit),
            'confidence': float(confidence),
            'predictions': prediction.tolist()[0]
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return {'error': str(e)}

def preprocess_for_training(img_data):
    """Preprocess image data for training"""
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Resize to 28x28 (MNIST format)
        img_resized = cv2.resize(img, (28, 28))
        
        # Invert if needed (assuming white digit on black background)
        if np.mean(img_resized) > 127:
            img_resized = 255 - img_resized
            
        # Normalize and reshape for CNN
        processed_img = img_resized.astype('float32') / 255
        processed_img = np.expand_dims(processed_img, axis=-1)  # Add channel dimension
        
        return processed_img
    except Exception as e:
        print(f"Error preprocessing image for training: {e}")
        return None

def handle_client_connection(conn, addr):
    """Handle individual client connection"""
    print(f"New connection from {addr}")
    client_id = addr[0]
    client_frames[client_id] = None
    client_predictions[client_id] = None
    
    buffer = b''
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
                
            buffer += data
            
            # Find complete frames
            while True:
                # Find JPEG start and end markers
                start = buffer.find(b'\xff\xd8')
                if start == -1:
                    break
                    
                end = buffer.find(b'\xff\xd9', start)
                if end == -1:
                    break
                    
                # Extract frame and update buffer
                frame_data = buffer[start:end+2]
                buffer = buffer[end+2:]
                
                # Decode frame
                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Store the frame
                    client_frames[client_id] = img.copy()
                    
                    # Process the frame if model is available
                    if model is not None:
                        # Normalize and reshape for CNN
                        processed_img = img.astype('float32') / 255
                        processed_img = np.expand_dims(processed_img, axis=-1)
                        processed_img = np.expand_dims(processed_img, axis=0)
                        
                        # Make prediction
                        prediction = model.predict(processed_img, verbose=0)
                        predicted_digit = np.argmax(prediction)
                        confidence = np.max(prediction) * 100
                        
                        # Store prediction
                        client_predictions[client_id] = {
                            'digit': int(predicted_digit),
                            'confidence': float(confidence),
                            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
                        }
                        
                        print(f"Client {client_id}: Predicted {predicted_digit} with confidence {confidence:.2f}%")
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        print(f"Connection closed from {addr}")
        conn.close()
        # Clean up client data
        if client_id in client_frames:
            del client_frames[client_id]
        if client_id in client_predictions:
            del client_predictions[client_id]

def start_tcp_server():
    """Start TCP server for receiving camera frames"""
    global tcp_server
    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_server.bind(('192.168.10.10', tcp_port))
    tcp_server.listen(5)
    
    print(f"TCP server listening on port {tcp_port}")
    
    while True:
        try:
            conn, addr = tcp_server.accept()
            client_thread = threading.Thread(target=handle_client_connection, args=(conn, addr))
            client_thread.daemon = True
            client_thread.start()
        except Exception as e:
            print(f"TCP server error: {e}")
            break

def generate_video_feed(client_id):
    """Generate frames for the video feed"""
    while True:
        # Get the latest frame for the client
        frame = client_frames.get(client_id)
        prediction = client_predictions.get(client_id)
        
        if frame is not None:
            # Create a frame with prediction overlay
            # Convert to color for adding text
            frame_with_text = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            if prediction:
                digit = prediction['digit']
                confidence = prediction['confidence']
                text = f"Digit: {digit} ({confidence:.1f}%)"
                cv2.putText(frame_with_text, text, (10, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame_with_text)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        # Short delay
        import time
        time.sleep(0.1)

@app.route('/')
def index():
    """Main page with video feeds from all clients and training interface"""
    # Generate HTML with video feeds for all connected clients
    clients_html = ""
    for client_id in client_frames.keys():
        clients_html += f"""
        <div class="client-feed">
            <h3>Client: {client_id}</h3>
            <img src="/video_feed/{client_id}" width="280" height="280">
        </div>
        """
    
    # Training interface HTML
    training_html = """
    <div class="training-section">
        <h2>Train with New Data</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="digitSelect">Select Digit:</label>
                <select id="digitSelect" name="digit" required>
                    <option value="">-- Select Digit --</option>
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="5">5</option>
                    <option value="6">6</option>
                    <option value="7">7</option>
                    <option value="8">8</option>
                    <option value="9">9</option>
                </select>
            </div>
            <div class="form-group">
                <label for="imageUpload">Upload Images:</label>
                <input type="file" id="imageUpload" name="images" accept=".png,.jpg,.jpeg" multiple required>
                <small>Select multiple images of the same digit</small>
            </div>
            <button type="submit" id="uploadBtn">Add to Training Set</button>
        </form>
        
        <div id="uploadStatus" class="status-box"></div>
        
        <div class="training-data">
            <h3>Current Training Data</h3>
            <div id="trainingStats">
                Loading training statistics...
            </div>
        </div>
        
        <div class="retrain-section">
            <h3>Retrain Model</h3>
            <button id="retrainBtn" onclick="retrainModel()">Retrain Model with New Data</button>
            <div id="retrainStatus" class="status-box"></div>
        </div>
    </div>
    """
    
    return f"""
    <html>
      <head>
        <title>Live Digit Recognition Server</title>
        <style>
          body {{ font-family: Arial; text-align: center; margin: 20px; }}
          .client-feed {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
          img {{ max-width: 100%; }}
          .dashboard {{ margin: 20px 0; padding: 10px; background: #f0f0f0; }}
          .tabs {{ display: flex; justify-content: center; margin: 20px 0; }}
          .tab {{ padding: 10px 20px; background: #ddd; cursor: pointer; margin: 0 5px; border-radius: 5px 5px 0 0; }}
          .tab.active {{ background: #4CAF50; color: white; }}
          .tab-content {{ display: none; padding: 20px; border: 1px solid #ddd; }}
          .tab-content.active {{ display: block; }}
          .form-group {{ margin-bottom: 15px; }}
          label {{ display: block; margin-bottom: 5px; }}
          select, input[type="file"] {{ width: 100%; padding: 8px; margin-bottom: 5px; }}
          button {{ 
            background-color: #4CAF50; 
            border: none; 
            color: white; 
            padding: 10px 20px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px; 
            margin: 4px 2px; 
            cursor: pointer; 
            border-radius: 4px; 
          }}
          button:hover {{ background-color: #45a049; }}
          .status-box {{ 
            margin: 15px 0; 
            padding: 10px; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            text-align: left; 
            min-height: 50px; 
          }}
          .training-section {{ max-width: 600px; margin: 0 auto; text-align: left; }}
          .training-data {{ margin-top: 20px; }}
          .retrain-section {{ margin-top: 20px; }}
          .digit-count {{ display: inline-block; margin: 5px; padding: 5px 10px; background: #f0f0f0; border-radius: 5px; }}
          .preview-container {{ 
            display: flex; 
            flex-wrap: wrap; 
            gap: 10px; 
            margin-top: 15px; 
          }}
          .image-preview {{ 
            width: 80px; 
            height: 80px; 
            border: 1px solid #ddd; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            position: relative; 
            background: #f9f9f9;
          }}
          .image-preview img {{ 
            max-width: 100%; 
            max-height: 100%; 
          }}
          .image-preview .remove-btn {{ 
            position: absolute; 
            top: -10px; 
            right: -10px; 
            background: red; 
            color: white; 
            border-radius: 50%; 
            width: 20px; 
            height: 20px; 
            line-height: 20px; 
            text-align: center; 
            cursor: pointer; 
            font-size: 12px; 
            font-weight: bold; 
          }}
        </style>
        <script>
          document.addEventListener('DOMContentLoaded', function() {{
            updateTrainingStats();
            
            // Tab functionality
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {{
              tab.addEventListener('click', function() {{
                // Remove active class from all tabs
                tabs.forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                this.classList.add('active');
                
                // Hide all tab content
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Show the corresponding tab content
                const tabId = this.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
              }});
            }});
            
            // Handle file upload form submission
            document.getElementById('uploadForm').addEventListener('submit', function(e) {{
              e.preventDefault();
              
              const formData = new FormData();
              const digit = document.getElementById('digitSelect').value;
              const files = document.getElementById('imageUpload').files;
              
              if (!digit) {{
                alert('Please select a digit');
                return;
              }}
              
              if (files.length === 0) {{
                alert('Please select at least one image');
                return;
              }}
              
              formData.append('digit', digit);
              for (let i = 0; i < files.length; i++) {{
                formData.append('images', files[i]);
              }}
              
              const statusDiv = document.getElementById('uploadStatus');
              statusDiv.innerHTML = 'Uploading images...';
              
              fetch('/upload_training_data', {{
                method: 'POST',
                body: formData
              }})
              .then(response => response.json())
              .then(data => {{
                if (data.success) {{
                  statusDiv.innerHTML = `Successfully added ${{data.count}} images for digit ${{digit}}`;
                  document.getElementById('uploadForm').reset();
                  updateTrainingStats();
                }} else {{
                  statusDiv.innerHTML = `Error: ${{data.error}}`;
                }}
              }})
              .catch(error => {{
                statusDiv.innerHTML = `Error: ${{error.message}}`;
              }});
            }});
            
            // Preview selected images
            document.getElementById('imageUpload').addEventListener('change', function() {{
              const previewContainer = document.createElement('div');
              previewContainer.className = 'preview-container';
              
              // Clear previous previews
              const oldPreview = document.querySelector('.preview-container');
              if (oldPreview) oldPreview.remove();
              
              const files = this.files;
              if (files.length > 0) {{
                for (let i = 0; i < files.length; i++) {{
                  const file = files[i];
                  if (!file.type.match('image.*')) continue;
                  
                  const reader = new FileReader();
                  reader.onload = function(e) {{
                    const previewDiv = document.createElement('div');
                    previewDiv.className = 'image-preview';
                    
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    previewDiv.appendChild(img);
                    
                    previewContainer.appendChild(previewDiv);
                  }};
                  
                  reader.readAsDataURL(file);
                }}
                
                // Insert preview after file input
                this.parentNode.appendChild(previewContainer);
              }}
            }});
          }});
          
          function updateTrainingStats() {{
            fetch('/training_stats')
            .then(response => response.json())
            .then(data => {{
              const statsDiv = document.getElementById('trainingStats');
              let html = '<p>Images per digit:</p><div>';
              
              for (let digit = 0; digit <= 9; digit++) {{
                const count = data.counts[digit] || 0;
                html += `<span class="digit-count">Digit ${{digit}}: ${{count}} images</span>`;
              }}
              
              html += '</div>';
              html += `<p>Total training images: ${{data.total}}</p>`;
              
              statsDiv.innerHTML = html;
            }})
            .catch(error => {{
              document.getElementById('trainingStats').innerHTML = `Error loading stats: ${{error.message}}`;
            }});
          }}
          
          function retrainModel() {{
            const statusDiv = document.getElementById('retrainStatus');
            statusDiv.innerHTML = 'Retraining model... This may take a few moments.';
            
            fetch('/retrain_model', {{
              method: 'POST'
            }})
            .then(response => response.json())
            .then(data => {{
              if (data.success) {{
                statusDiv.innerHTML = `Model successfully retrained! Accuracy: ${{data.accuracy.toFixed(4)}}`;
              }} else {{
                statusDiv.innerHTML = `Error: ${{data.error}}`;
              }}
            }})
            .catch(error => {{
              statusDiv.innerHTML = `Error: ${{error.message}}`;
            }});
          }}
        </script>
      </head>
      <body>
        <h1>Live Digit Recognition Server</h1>
        <div class="dashboard">
          <p>Connected clients: {len(client_frames)}</p>
          <p>Model status: {'Loaded' if model is not None else 'Not loaded'}</p>
        </div>
        
        <div class="tabs">
          <div class="tab active" data-tab="tab-clients">Live Clients</div>
          <div class="tab" data-tab="tab-training">Training</div>
        </div>
        
        <div id="tab-clients" class="tab-content active">
          <div class="clients">
            {clients_html if clients_html else "<p>No clients connected</p>"}
          </div>
        </div>
        
        <div id="tab-training" class="tab-content">
          {training_html}
        </div>
      </body>
    </html>
    """

@app.route('/video_feed/<client_id>')
def video_feed(client_id):
    """Video feed endpoint for a specific client"""
    if client_id in client_frames:
        return Response(generate_video_feed(client_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Client not found", 404

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """Handle training data upload"""
    if 'images' not in request.files:
        return jsonify({'success': False, 'error': 'No images provided'})
    
    if 'digit' not in request.form:
        return jsonify({'success': False, 'error': 'No digit label provided'})
    
    try:
        digit = int(request.form['digit'])
        if digit < 0 or digit > 9:
            return jsonify({'success': False, 'error': 'Digit must be between 0 and 9'})
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid digit label'})
    
    files = request.files.getlist('images')
    count = 0
    for file in files:
        if file and allowed_file(file.filename):
            # Create digit subfolder if it doesn't exist
            digit_folder = os.path.join(UPLOAD_FOLDER, str(digit))
            if not os.path.exists(digit_folder):
                os.makedirs(digit_folder)
            
            # Save the file
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(digit_folder, unique_filename)
            file.save(file_path)
            
            # Preprocess and save for training
            with open(file_path, 'rb') as f:
                img_data = f.read()
                processed_img = preprocess_for_training(img_data)
                
                if processed_img is not None:
                    # Add to training data
                    global training_images, training_labels
                    training_images.append(processed_img)
                    training_labels.append(digit)
                    count += 1
    
    return jsonify({
        'success': True,
        'count': count,
        'digit': digit
    })

@app.route('/training_stats')
def training_stats():
    """Get statistics about the training data"""
    stats = {
        'counts': {},
        'total': 0
    }
    
    # Count from training_labels
    for label in training_labels:
        if label not in stats['counts']:
            stats['counts'][label] = 0
        stats['counts'][label] += 1
        stats['total'] += 1
    
    # Also count from the filesystem
    for digit in range(10):
        digit_folder = os.path.join(UPLOAD_FOLDER, str(digit))
        if os.path.exists(digit_folder):
            files = [f for f in os.listdir(digit_folder) if os.path.isfile(os.path.join(digit_folder, f)) and allowed_file(f)]
            if digit not in stats['counts']:
                stats['counts'][digit] = 0
            stats['counts'][digit] += len(files)
            stats['total'] += len(files)
    
    return jsonify(stats)

@app.route('/retrain_model', methods=['POST'])
def retrain_model_endpoint():
    """Retrain the model with the new training data"""
    global model, training_images, training_labels
    
    # Check if training data exists using length
    if len(training_images) == 0 or len(training_labels) == 0:
        # Load from filesystem if no in-memory data
        loaded_images = []
        loaded_labels = []
        
        for digit in range(10):
            digit_folder = os.path.join(UPLOAD_FOLDER, str(digit))
            if os.path.exists(digit_folder):
                files = [f for f in os.listdir(digit_folder) if os.path.isfile(os.path.join(digit_folder, f)) and allowed_file(f)]
                for file in files:
                    file_path = os.path.join(digit_folder, file)
                    with open(file_path, 'rb') as f:
                        img_data = f.read()
                        processed_img = preprocess_for_training(img_data)
                        if processed_img is not None:
                            loaded_images.append(processed_img)
                            loaded_labels.append(digit)
        
        if len(loaded_images) == 0:
            return jsonify({
                'success': False,
                'error': 'No training data available'
            })
        
        training_images = loaded_images
        training_labels = loaded_labels
    
    # Convert lists to numpy arrays
    training_images_np = np.array(training_images)
    training_labels_np = np.array(training_labels)
    
    print(f"Retraining model with {len(training_images)} images...")
    
    try:
        if model is None:
            model = load_or_train_model()
        
        # Check the loss function used in the model
        loss_function = model.loss
        
        # Use sparse categorical crossentropy for integer labels
        # and categorical crossentropy for one-hot encoded labels
        if 'sparse' in str(loss_function).lower():
            # Use integer labels directly
            retrain_model(model, training_images_np, training_labels_np)
        else:
            # Convert labels to categorical format
            training_labels_cat = tf.keras.utils.to_categorical(training_labels_np, 10)
            retrain_model(model, training_images_np, training_labels_cat)
        
        # Create a small validation set
        val_size = min(len(training_images_np), 100)
        val_indices = np.random.choice(len(training_images_np), val_size, replace=False)
        val_images = training_images_np[val_indices]
        val_labels = training_labels_np[val_indices]
        
        # Evaluate model - use the appropriate format based on loss function
        if 'sparse' in str(loss_function).lower():
            loss, accuracy = model.evaluate(val_images, val_labels, verbose=0)
        else:
            val_labels_cat = tf.keras.utils.to_categorical(val_labels, 10)
            loss, accuracy = model.evaluate(val_images, val_labels_cat, verbose=0)
        
        return jsonify({
            'success': True,
            'accuracy': float(accuracy),
            'loss': float(loss),
            'num_samples': len(training_images)
        })
    except Exception as e:
        print(f"Error retraining model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for digit prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    img_data = image_file.read()
    
    result = process_digit_image(img_data)
    return jsonify(result)

@app.route('/model')
def serve_model():
    """Serve the CNN model file"""
    model_path = 'cnn_model.keras'
    if os.path.exists(model_path):
        return send_file(model_path,
                        mimetype='application/octet-stream',
                        as_attachment=True,
                        download_name='cnn_model.keras')
    return "Model not found", 404

@app.route('/status')
def status():
    """Check if server is running and model is loaded"""
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "connected_clients": list(client_frames.keys()),
        "client_count": len(client_frames),
        "training_data_count": len(training_images)
    })

def main():
    """Main function to start the server"""
    print("Initializing Digit Recognition Server...")
    
    # Initialize the model
    print("Loading CNN model...")
    if not initialize_model():
        print("Warning: Failed to load model")
    
    # Start TCP server in a separate thread
    tcp_thread = threading.Thread(target=start_tcp_server)
    tcp_thread.daemon = True
    tcp_thread.start()
    
    print("Starting Flask web server...")
    app.run(host='192.168.10.10', port=8000, threaded=True)

if __name__ == '__main__':
    main()