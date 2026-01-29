import os, time, io, base64
import numpy as np
import msgpack
import msgpack_numpy as m
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2

m.patch()
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

def fast_encode_img(image_np, target_size=(400, 300)):
    if image_np is None: return None
    try:
        img_resized = cv2.resize(image_np.astype('uint8'), 
                            target_size, 
                            interpolation=cv2.INTER_NEAREST)
        _, buffer = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return base64.b64encode(buffer).decode('utf-8')
    except: return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    start_t = time.time()
    try:
        data = msgpack.unpackb(request.data, raw=False)
        payload = {
            'images': {
                'top': fast_encode_img(data.get('image_top')),
                'left': fast_encode_img(data.get('image_left')),
                'right': fast_encode_img(data.get('image_right'))
            },
            'telemetry': data.get('telemetry').tolist(),
            'start_idx': data.get('start_idx'),
            'step_idx': data.get('step_idx'),
            'latency': int((time.time() - start_t) * 1000)
        }
        socketio.emit('new_frame', payload)
        return 'OK', 200
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)