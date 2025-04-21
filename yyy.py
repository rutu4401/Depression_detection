import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib  # For saving and loading the model

# Step 1: Load the CSV file
file_path = "synthetic_depression_data.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Step 2: Encode the target variable
label_encoder = LabelEncoder()
data["Depression_Level"] = label_encoder.fit_transform(data["Depression_Level"])

# Step 3: Prepare the input (X) and target (y)
X = data.drop(columns=["Depression_Level", "Total_Score"])  # Drop the target and any non-feature columns
y = data["Depression_Level"]

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Save the trained model
model_filename = "depression_model.joblib"
label_encoder_filename = "label_encoder.joblib"
joblib.dump(model, model_filename)
joblib.dump(label_encoder, label_encoder_filename)
print(f"Model saved as {model_filename}")
print(f"Label encoder saved as {label_encoder_filename}")

# Step 8: Define the questionnaire and scoring
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

# Collect user responses
def collect_responses():
    responses = []
    for question in questions:
        print(question)
        answer = input("Choose (Never, Sometimes, Often, Almost always): ")
        responses.append(answer_map.get(answer, 0))
    return responses

# Step 9: Load the model and make predictions
def predict_depression():
    loaded_model = joblib.load(model_filename)
    loaded_label_encoder = joblib.load(label_encoder_filename)

    responses = collect_responses()
    prediction = loaded_model.predict([responses])[0]
    predicted_label = loaded_label_encoder.inverse_transform([prediction])[0]
    print("Predicted Depression Level:", predicted_label)

# Uncomment the line below to predict depression level after saving the model
# predict_depression()
