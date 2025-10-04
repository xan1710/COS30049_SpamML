# Logistic Regression Spam Classification
# Author: Your Name
# Date: 2025-09-30


import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from preprocessing import load_dataset, clean_text, extract_features

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

df = load_dataset()
if df is None:
    raise ValueError("No data found. Run preprocessing first.")
print(f"📊 {len(df):,} samples, {df['label'].mean():.1%} spam")

# Receive and preprocess features
X_text = df['text'].fillna('')
y = df['label']

# Feature preparation - Essential feature 1: Combined TF-IDF + engineered features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X_text)

# Extract spam features for each text
spam_features = [extract_features(str(text)) + [len(str(text)), len(str(text).split())] 
                for text in X_text]
X_features = np.array(spam_features)
print(f"Extracted engineered features shape: {X_features.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model - Essential feature 2: Optimized LogisticRegression
model = LogisticRegression(C=10, random_state=42, max_iter=20000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"🎯 AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Save model and vectorizer
joblib.dump({'model': model, 'vectorizer': vectorizer}, MODEL_DIR / 'log_reg_model.joblib')
print("💾 Logistic Regression Model saved")

# Test predictions
def predict_email(text):
    clean_text_input = clean_text(text)
    if not clean_text_input:
        return "Error: Empty text"
    
    X = vectorizer.transform([clean_text_input])
    y_prob = model.predict_proba(X)[0, 1]
    label = 'Spam' if y_prob >= 0.5 else 'Ham'
    return label, y_prob

# Example usage
test_emails = [
    "FREE MONEY! Click here to win $10000 now! Limited time offer!",
    "Hi John, let's meet for lunch tomorrow at 12pm. Looking forward to seeing you!",
    "Dear user, you have won a prize! claim now verification required",
    "Meeting agenda attached for project update discussion", 
    "Urgent: verify your bank account immediately or account will be closed",
    "Lunch plans? Let me know your preference for the restaurant",
    "Free gift card available: update your personal info to claim",
    "Congratulations! You've won $1 million dollars! Click to claim now!",
    "Your account needs verification. Click here now before midnight!",
    "Team meeting scheduled for tomorrow at 3pm in conference room A",
    "Invoice attached for your recent purchase from our store",
    "Limited time offer - act now to save 50% money back guarantee!"
]

for example_email in test_emails:
    label, prob = predict_email(example_email)
    print(f"Prediction: {label} (Probability: {prob:.2f}) - {example_email:.15}")