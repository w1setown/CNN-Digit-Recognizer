from flask import Flask, Response
import subprocess
import threading
import time
import socket

app = Flask(__name__)

# Global variables
rpicam_process = None
tcp_port = 8001  # Port for rpicam to stream to
latest_frame = None
frame_lock = threading.Lock()

def start_rpicam_stream():
    global rpicam_process
    
    # Start rpicam-vid to output to TCP
    rpicam_process = subprocess.Popen([
        "rpicam-vid",
        "-t", "0",           # Run indefinitely
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "--codec", "mjpeg",
        "--inline",          # Important for MJPEG streaming
        "--listen",          # Listen for connections
        "-o", f"tcp://172.16.3.113:{tcp_port}"  # Listen on all interfaces
    ])
    
    # Give rpicam time to start
    time.sleep(2)
    
    # Start frame capture thread
    frame_thread = threading.Thread(target=capture_frames)
    frame_thread.daemon = True
    frame_thread.start()

def capture_frames():
    global latest_frame
    
    # Connect to the TCP server that rpicam-vid created
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', tcp_port))
    
    try:
        # Read data from the socket
        buffer = b''
        while True:
            data = s.recv(4096)
            if not data:
                break
                
            buffer += data
            
            # Find JPEG frame boundaries
            while True:
                start = buffer.find(b'\xff\xd8')  # JPEG start marker
                if start == -1:
                    break
                    
                end = buffer.find(b'\xff\xd9', start)  # JPEG end marker
                if end == -1:
                    break
                    
                # Extract the JPEG frame
                frame = buffer[start:end+2]
                buffer = buffer[end+2:]
                
                # Update the latest frame with thread safety
                with frame_lock:
                    latest_frame = frame
    finally:
        s.close()

def stop_rpicam_stream():
    global rpicam_process
    if rpicam_process:
        rpicam_process.terminate()
        rpicam_process = None

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Raspberry Pi Camera Stream (rpicam)</title>
        <style>
          body { font-family: Arial; text-align: center; margin-top: 50px; }
          img { max-width: 100%; }
          .controls { margin: 20px 0; padding: 10px; background: #f0f0f0; }
        </style>
      </head>
      <body>
        <h1>Raspberry Pi Camera Stream</h1>
        <div class="controls">
          <p>Using rpicam-vid for streaming - Multiple client support</p>
        </div>
        <img src="/stream" />
      </body>
    </html>
    """

def generate_frames():
    global latest_frame
    
    while True:
        # Get the latest frame with thread safety
        with frame_lock:
            frame = latest_frame
            
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # If no frame available, provide a simple empty frame
            empty_frame = b'\xff\xd8\xff\xdb\x00\x84\x00\x10\x0b\x0c\x0e\x0c\x0a\x10\x0e\x0d\x0e\x12\x11\x10\x13\x18\x28\x1a\x18\x16\x16\x18\x31\x23\x25\x1d\x28\x3a\x33\x3d\x3c\x39\x33\x38\x37\x40\x48\x5c\x4e\x40\x44\x57\x45\x37\x38\x50\x6d\x51\x57\x5f\x62\x67\x68\x67\x3e\x4d\x71\x79\x70\x64\x78\x5c\x65\x67\x63\x01\x11\x12\x12\x18\x15\x18\x2f\x1a\x1a\x2f\x63\x42\x38\x42\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\x63\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x15\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\x00\x00\xff\xd9'
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + empty_frame + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the rpicam stream in a separate thread
    stream_thread = threading.Thread(target=start_rpicam_stream)
    stream_thread.daemon = True
    stream_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=8000, threaded=True)
    finally:
        stop_rpicam_stream()
