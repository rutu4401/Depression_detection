import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("new_dataset.csv")  # Ensure the dataset is in the same folder

# Convert categorical responses to numerical values
response_mapping = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4
}

# Apply mapping to question responses
for col in df.columns[1:-1]:  # Exclude 'User ID' and 'Depression Level'
    df[col] = df[col].map(response_mapping)

# Encode target labels (Depression Level)
label_encoder = LabelEncoder()
df["Depression Level"] = label_encoder.fit_transform(df["Depression Level"])

# Split data into features (X) and target labels (y)
X = df.drop(columns=["User ID", "Depression Level"])
y = df["Depression Level"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions & Evaluation (XGBoost)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
classification_rep_xgb = classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_)

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib

# Define parameter grid for XGBoost
param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [8, 10, 12],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Perform GridSearchCV to find best parameters
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False)
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_xgb_model = grid_search.best_estimator_

# Predictions & Evaluation (Tuned XGBoost)
y_pred_xgb_tuned = best_xgb_model.predict(X_test)
accuracy_xgb_tuned = accuracy_score(y_test, y_pred_xgb_tuned)
classification_rep_xgb_tuned = classification_report(y_test, y_pred_xgb_tuned, target_names=label_encoder.classes_)

# Print Accuracy & Report
print(f"Tuned XGBoost Accuracy: {accuracy_xgb_tuned:.2f}")
print("\nClassification Report:")
print(classification_rep_xgb_tuned)

# Save best model
joblib.dump(best_xgb_model, "xgboost_depression_model_tuned.pkl")
print("Tuned model saved as xgboost_depression_model_tuned.pkl")

