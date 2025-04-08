from flask import Flask, Response
import subprocess
import threading
import time
import socket

app = Flask(__name__)

# Global variables
rpicam_process = None
tcp_port = 8001  # Port for rpicam to stream to

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
        "-o", f"tcp://0.0.0.0:{tcp_port}"
    ])
    
    # Give rpicam time to start
    time.sleep(2)

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
          <p>Using rpicam-vid for streaming</p>
        </div>
        <img src="/stream" />
      </body>
    </html>
    """

def generate_frames():
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
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        s.close()

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
