from flask import Flask, Response
from picamera2 import Picamera2  # type: ignore
import cv2
import threading

app = Flask(__name__)
picam2 = Picamera2()
picam2.start()

def gen_frames():
    while True:
        frame = picam2.capture_array()
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        jpg_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
      <head>
        <title>Raspberry Pi Camera Stream</title>
      </head>
      <body>
        <h1>Live Camera Stream</h1>
        <img src="/video_feed" width="640" height="480" />
      </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
