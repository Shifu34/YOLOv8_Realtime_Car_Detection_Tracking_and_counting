import main
import predict
import json

import threading

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

from socket_manager import socketio

import cv2


# Initialize the Flask app and set a secret key for session security
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'


# Global flag to track if the thread is already running
is_thread_running = False

# Function to call the prediction function in a separate thread
def predict_inthread():
    main.predict()  # Assuming predict() is the function for the detection process

# Create a thread to run the prediction function in parallel
function_thread = threading.Thread(target=predict_inthread)

# Route for the main page (index.html)
@app.route('/')
def index():
    return render_template('index.html')  # Renders the HTML template for the main page

# SocketIO event handler for client connection
@socketio.on('connect')
def on_connect():
    global is_thread_running

    # Start the thread if it's not running yet
    if not is_thread_running:
        function_thread.daemon = True
        function_thread.start()
        is_thread_running = True


if __name__ == '__main__':
    # Initialize SocketIO with the Flask app and run the application in debug mode
    socketio.init_app(app)
    socketio.run(app, debug=True)