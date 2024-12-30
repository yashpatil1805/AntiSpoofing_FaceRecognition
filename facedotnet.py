import cv2
import dlib
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load face detector and face recognizer models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download it from dlib
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  # Download this model

# Function to detect faces and return encodings
def get_face_encodings(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    encodings = []
    for face in faces:
        shape = sp(gray, face)
        encoding = facerec.compute_face_descriptor(image, shape)
        encodings.append(np.array(encoding))
    return encodings

@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    # Get the image from the request
    file = request.files['file']
    img = Image.open(file)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Get face encodings
    encodings = get_face_encodings(img)

    if len(encodings) == 0:
        return jsonify({'message': 'No face detected'}), 400
    else:
        return jsonify({'message': 'Face recognized', 'encodings': encodings[0].tolist()}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Expose it locally
