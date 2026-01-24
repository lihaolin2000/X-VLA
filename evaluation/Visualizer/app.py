import os
import numpy as np
from flask_cors import CORS
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import io
import base64
import msgpack
import msgpack_numpy as m
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")
# Configure SocketIO with CORS
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for WebSocket connections

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Apply msgpack_numpy patch
m.patch()

# Create video frame and encode as base64
def generate_video_frame(image_np):
    image = Image.fromarray(image_np)  # Convert numpy array to image
    img_stream = io.BytesIO()
    image.save(img_stream, format='PNG')  # Save image to memory stream
    img_stream.seek(0)
    encoded_image = base64.b64encode(img_stream.read()).decode('utf-8')  # Base64 encode image
    return encoded_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_data():
    print("Received POST request")
    data = msgpack.unpackb(request.data, raw=False)  # Unpack msgpack data
    try:
        print("Processing image")
        
        # Process image
        frame_data = generate_video_frame(data['image'])
        
        # Prepare data to send via WebSocket
        data_to_send = {'frame': frame_data}
        
        # Emit the data to the front-end via WebSocket
        socketio.emit('new_frame', data_to_send)
        print("Data sent via WebSocket")
        return 'Data received successfully', 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return f'Error: {str(e)}', 500

if __name__ == '__main__':
    # Path to SSL certificate and private key files
    # cert_file = 'server.crt'  # 替换为你自己的证书文件路径
    # key_file = 'server.key'   # 替换为你自己的私钥文件路径

    # Run the app with SSL
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)
