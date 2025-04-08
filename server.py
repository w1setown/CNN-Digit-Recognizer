from flask import Flask, Response, jsonify, send_file, request
import socket
import threading
import numpy as np
import cv2
import os
import io
from PIL import Image
import tensorflow as tf
from model import load_or_train_model

app = Flask(__name__)

# Global variables
model = None
tcp_server = None
tcp_port = 8001
client_frames = {}  # Store frames from connected clients
client_predictions = {}  # Store predictions for connected clients

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
    tcp_server.bind(('0.0.0.0', tcp_port))
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
    """Main page with video feeds from all clients"""
    # Generate HTML with video feeds for all connected clients
    clients_html = ""
    for client_id in client_frames.keys():
        clients_html += f"""
        <div class="client-feed">
            <h3>Client: {client_id}</h3>
            <img src="/video_feed/{client_id}" width="280" height="280">
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
        </style>
        <meta http-equiv="refresh" content="10">
      </head>
      <body>
        <h1>Live Digit Recognition Server</h1>
        <div class="dashboard">
          <p>Connected clients: {len(client_frames)}</p>
          <p>Model status: {'Loaded' if model is not None else 'Not loaded'}</p>
        </div>
        <div class="clients">
          {clients_html if clients_html else "<p>No clients connected</p>"}
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
        "client_count": len(client_frames)
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
    app.run(host='0.0.0.0', port=8000, threaded=True)

if __name__ == '__main__':
    main()