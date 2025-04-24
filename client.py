import socket
import cv2
import numpy as np
import requests
import time
import io
from PIL import Image
import threading

# Server connection details
SERVER_HOST = '192.168.10.1'
SERVER_PORT = 8000
MODEL_ENDPOINT = f'http://{SERVER_HOST}:{SERVER_PORT}/predict'
MODEL_FILE_ENDPOINT = f'http://{SERVER_HOST}:{SERVER_PORT}/model'

# For streaming video to server
TCP_PORT = 8001

# Global variables
camera = None
stream_active = False
stream_thread = None

def initialize_camera():
    """Initialize the Raspberry Pi camera"""
    global camera
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            return False
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def preprocess_frame(frame):
    """Preprocess frame for digit recognition"""
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Find the digit in the frame (assuming it's centered)
    height, width = gray.shape
    center_size = min(height, width) // 2
    center_x, center_y = width // 2, height // 2
    
    # Extract the region of interest (ROI)
    roi = gray[center_y - center_size:center_y + center_size,
               center_x - center_size:center_x + center_size]
    
    # Resize to 28x28 (MNIST format)
    roi_resized = cv2.resize(roi, (28, 28))
    
    # Invert if neede
    if np.mean(roi_resized) > 127:
        roi_resized = 255 - roi_resized
        
    return roi_resized

def download_model():
    """Download the CNN model from the server"""
    try:
        print(f"Downloading model from {MODEL_FILE_ENDPOINT}")
        response = requests.get(MODEL_FILE_ENDPOINT, stream=True)
        
        if response.status_code == 200:
            with open('cnn_model.keras', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully")
            return True
        else:
            print(f"Failed to download model: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def stream_to_server():
    """Stream video frames to server via TCP socket"""
    global stream_active
    stream_active = True
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, TCP_PORT))
    
    try:
        while stream_active and camera.isOpened():
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame")
                break

            display_frame = frame.copy()
            height, width = frame.shape[:2]
            center_size = min(height, width) // 2
            center_x, center_y = width // 2, height // 2
            cv2.rectangle(display_frame, 
                        (center_x - center_size, center_y - center_size),
                        (center_x + center_size, center_y + center_size),
                        (0, 255, 0), 2)
            cv2.imshow('Camera Feed', display_frame)
            
            # Process frame and send to server
            roi = preprocess_frame(frame)
            
            # Convert to JPEG and send
            _, img_encoded = cv2.imencode('.jpg', roi)
            sock.sendall(img_encoded.tobytes())
            
            # Send frame delimiter
            sock.sendall(b'\xff\xd9')
            
            # Brief delay
            time.sleep(0.1)
            
            # Check for key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Streaming error: {e}")
    finally:
        sock.close()
        stream_active = False

def start_stream():
    """Start the streaming thread"""
    global stream_thread
    if stream_thread is None or not stream_thread.is_alive():
        stream_thread = threading.Thread(target=stream_to_server)
        stream_thread.daemon = True
        stream_thread.start()
        print("Streaming started")

def stop_stream():
    """Stop the streaming thread"""
    global stream_active
    stream_active = False
    print("Streaming stopped")

def check_server_status():
    """Check if the server is online and ready"""
    try:
        response = requests.get(f'http://{SERVER_HOST}:{SERVER_PORT}/status')
        return response.status_code == 200 and response.json().get('status') == 'running'
    except:
        return False

def main():
    """Main function to run the client"""
    print("Initializing Raspberry Pi digit recognition client...")
    
    # Initialize camera
    if not initialize_camera():
        print("Failed to initialize camera. Exiting.")
        return
    
    # Check server status
    print("Checking server status...")
    if not check_server_status():
        print("Server is not available. Please start the server first.")
        return
    
    # Start streaming
    print("Starting video stream...")
    start_stream()
    
    print("Client is running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down client...")
    finally:
        stop_stream()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()