***🧠 Real-Time Depression Motion Detection System using CNN & XGBoost***
This is a Flask-based real-time Depression Detection System that uses a CNN model to detect facial expressions from camera feed and predicts depression based on user responses using an XGBoost ML model.

***✨ Features***
📸 Real-time facial emotion detection using CNN (emotion.h5)

🧾 Questionnaire-based depression prediction using XGBoost (xgb_model.joblib)

🧠 Emotion count and depression result displayed on a separate result page

🧩 Combines Deep Learning + Machine Learning models

🎯 Separate borders for camera and questionnaire section

💻 Flask Web Interface for user interaction

📊 Future scope: Add charts, animations, and better visualization


***💡 How It Works***
User opens the app and camera starts detecting real-time facial emotions.

The CNN model classifies emotions (Happy, Sad, Angry, etc.).

User fills out a depression questionnaire.

The XGBoost model uses the emotion count + answers to predict depression.

The result is shown on a final results page with emotion statistics.

***🧠 Tech Stack***
🧪 CNN (Keras) – For real-time emotion classification

📊 XGBoost – For depression classification using combined input

🌐 Flask – Lightweight Python web framework

🎥 OpenCV – Capturing camera feed

🧮 NumPy, Pandas – Data processing

📄 Jupyter Notebooks – For model training and experimentation

***📁 Project Structure***

depression_motion/
│
├── app.py                     # Flask app for running camera and questionnaire
├── emotion.py                 # Real-time emotion detection logic
├── modelgen.py                # ML model training and feature generation
├── deeppp.py                  # CNN model training script
├── emotion_depression.ipynb  # Final implementation notebook
├── working.ipynb             # Testing & experimentation notebook
│
├── templates/
│   ├── index.html             # Main interface (camera + questionnaire)
│   └── result.html            # Final results and emotion analysis
│
├── static/                    # CSS, JS, images (if any)
├── emotion.h5                # Trained CNN model for facial emotion
├── xgb_model.joblib          # Trained XGBoost model for depression
├── depression_dataset.csv    # Dataset used for training ML model
├── new_dataset.csv           # Processed dataset
├── result.csv                # Saved outputs

***⚙️ Setup Instructions***

*1.Clone the repository*

git clone https://github.com/rutu4401/depression-motion-detector.git
cd depression-motion-detector

*2.(Optional) Create a virtual environment*
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

*3.Install the required packages*
pip install -r requirements.txt

*4.Run the Flask app*
python app.py

*5.Open in browser*
http://127.0.0.1:5000/

***📊 Dataset***
Facial Emotion Dataset – Used to train the CNN model

Depression Questionnaire Dataset – Used to train XGBoost model

All datasets are included in .csv format

*👩‍💻 Developed By*
Rutuja Gaikwad

*🛡️ License*
This project is licensed under the MIT License.

