import cv2
import numpy as np
from transformers import pipeline
import torch

# Load the pre-trained emotion classification model
emotion_recognition = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Start capturing video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for easier processing and faster performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use Haar Cascade to detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each face detected in the webcam feed
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face and use it to predict emotion using text-based approach
        # You can also apply a facial emotion recognition model here, but for now, we will use a basic text pipeline
        face_image = frame[y:y + h, x:x + w]
        
        # Dummy text input based on your observation (serious vs not serious)
        # You can replace this with more advanced facial expression-based text or analysis
        if np.mean(face_image) > 100:  # This is just a placeholder logic for seriousness.
            emotion_text = "I'm serious."
        else:
            emotion_text = "I'm not serious."

        # Get emotion prediction from the Hugging Face model
        emotion = emotion_recognition(emotion_text)

        # Show the emotion prediction
        cv2.putText(frame, f"Emotion: {emotion[0]['label']}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
