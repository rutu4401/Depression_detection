from flask import Flask, render_template, request, jsonify, session
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import base64
from collections import Counter
from joblib import load

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load Emotion Detection Model
emotion_model = load_model("emotion_detection_model.h5")

# Emotion Labels (Based on Model Training)
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load Depression Model
depression_model = load("depression_model.joblib")
label_encoder = load("label_encoder.joblib")

# Questions and Answer Mapping
questions = [
    "How often have you felt down, depressed, or hopeless?",
    "How often have you experienced a lack of interest or pleasure in doing things?",
    "How frequently have you had trouble sleeping (too much or too little)?",
    "How often do you feel tired or have low energy?",
    "How often have you felt bad about yourself, or that you are a failure?",
    "How often have you had difficulty concentrating on tasks, such as reading or watching TV?",
    "How often have you felt anxious, worried, or on edge?",
    "How often do you feel that life isn’t worth living?",
    "How frequently do you feel irritable or easily frustrated?",
    "How often do you feel disconnected from people around you?"
]

answer_map = {
    "Never": 0,
    "Sometimes": 1,
    "Often": 2,
    "Almost always": 3
}

emotion_counts = Counter()  # Store detected emotions

# Function to Analyze Emotion from Image
def analyze_emotion_from_base64(image_data):
    global emotion_counts
    img_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # For grayscale
    img = np.expand_dims(img, axis=0)   # Add batch dimension

    try:
        preds = emotion_model.predict(img)
        print(f"Model Raw Predictions: {preds}")  # Debugging output
        emotion_idx = np.argmax(preds[0])
        dominant_emotion = emotion_labels[emotion_idx]
        print(f"Detected Emotion: {dominant_emotion}")  # Debugging output
        emotion_counts[dominant_emotion] += 1
    except Exception as e:
        print(f"Emotion detection error: {e}")

# Route for Real-Time Emotion Detection
@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    data = request.get_json()
    image_data = data['image']
    analyze_emotion_from_base64(image_data)
    print(f"Updated Emotion Counts: {emotion_counts}")  # Debugging
    return jsonify({"emotion_counts": dict(emotion_counts)})

# Main Route (Questionnaire + Camera)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        scores = [int(request.form.get(f"q{i}")) for i in range(len(questions))]
        prediction = depression_model.predict([scores])[0]
        result = label_encoder.inverse_transform([prediction])[0]

        # Save result in session to display on the next page
        session['result'] = result
        session['emotion_counts'] = dict(emotion_counts)

        return render_template("result.html", result=result, emotion_counts=emotion_counts)

    return render_template("index.html", questions=questions, answer_map=answer_map)

# Route for Result Page
@app.route("/result")
def result():
    result = session.get('result', "No result")
    emotion_counts = session.get('emotion_counts', {})

    return render_template("result.html", result=result, emotion_counts=emotion_counts)

# Start Flask Server
if __name__ == "__main__":
    app.run(debug=True)
