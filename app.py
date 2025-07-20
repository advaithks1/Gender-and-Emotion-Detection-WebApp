import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TensorFlow from allocating all GPU memory at once

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load Gender Detection Model
gender_model = load_model('gender_detection_model1.h5')

# Load Emotion Detection Model (EfficientNetB4)
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(7, activation='softmax')(x)
emotion_model = Model(inputs=base_model.input, outputs=output)
emotion_model.load_weights('emotion_weights_final.h5')

# Load Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion Labels
emotion_labels = {
    0: 'angry', 1: 'disgusted', 2: 'fearful',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'
}

# Preprocessing function
def preprocess_face(face):
    face = cv2.resize(face, (128, 128))
    face = img_to_array(face) / 255.0
    return np.expand_dims(face, axis=0)

# Prediction + drawing
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        pf = preprocess_face(face)

        # Predict gender
        gender_pred = gender_model.predict(pf)[0][0]
        gender = "Male" if gender_pred > 0.5 else "Female"
        g_perc = round(gender_pred * 100, 2) if gender == 'Male' else round((1 - gender_pred) * 100, 2)

        # Predict emotion
        emotion_pred = emotion_model.predict(pf)
        emotion = emotion_labels[np.argmax(emotion_pred)]

        # Annotate
        label = f"{gender} ({g_perc}%) | {emotion}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (36, 255, 12), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    return frame

# Stream video to browser
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = detect(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown')
def shutdown():
    camera.release()
    cv2.destroyAllWindows()
    return "Camera released."

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
