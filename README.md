:

🧠 Real-Time Depression Motion Detection System using Deep Learning & ML
This is a Flask-based real-time Depression Detection System that uses a trained CNN model for facial emotion recognition and an XGBoost model for mental health prediction based on questionnaire responses.

🚨 Features
🎥 Real-time facial expression detection using webcam

🧠 Depression prediction based on XGBoost ML model

📋 Interactive questionnaire for psychological evaluation

📊 Emotion count display with separate results page

🎯 Combines deep learning (CNN) and machine learning (XGBoost)

💻 User-friendly Flask Web Interface

✨ Custom UI with camera + questions shown in separate sections

💡 How It Works
User visits the web interface.

The camera captures facial expressions in real time.

Detected emotions are analyzed using a CNN model.

The user answers a set of questions in the questionnaire.

The emotion count + questionnaire responses are passed to the XGBoost model.

The app predicts the user's likelihood of being in a depressive state.

Final results and statistics are displayed on a separate results page.

🧠 Tech Stack
TensorFlow / Keras – Facial emotion detection model

XGBoost – Depression classification

Flask – Python web framework

OpenCV – Camera feed processing

Pandas, NumPy, Scikit-learn – Data preprocessing & manipulation

HTML/CSS (Jinja2) – Frontend templates

📁 Project Structure

depression_motion/
│
├── app.py                     # Main Flask app entry
├── emotion.py                 # Facial emotion detection logic
├── deeppp.py                  # Model architecture & training
├── modelgen.py                # Generates and trains ML models
├── emotion_depression.ipynb  # Final Jupyter notebook for testing
├── working.ipynb             # Experimentation notebook
├── *.pkl, *.joblib, *.h5      # Trained model files
├── *.csv                      # Dataset files
│
├── templates/
│   ├── index.html             # Main UI (camera + questionnaire)
│   └── result.html            # Emotion + depression result page
│
└── static/                    # Optional static files (CSS, JS, images)

⚙️ Setup Instructions
1.Clone the repository
git clone https://github.com/rutu4401/depression-motion-detector.git
cd depression-motion-detector

2.(Optional) Create a virtual environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


3.Install dependencies
pip install -r requirements.txt

4.Run the Flask app
python app.py

5.Access the application
http://127.0.0.1:5000/


📊 Datasets Used
depression_dataset.csv – For training depression classifier

new_dataset.csv – Cleaned/merged data for questionnaire model

Real-time webcam feed for emotion detection

👩‍💻 Developed By
Rutuja Gaikwad

🛡️ License
This project is licensed under the MIT License. See the LICENSE file for more details.
