# Simplified Logistic Regression Spam Classification Pipeline
# Author: Your Name
# Date: 2025-09-30

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib
from preprocessing import load_cleaned_data, clean_text, extract_features

# Load dataset
df = load_cleaned_data()
if df is None:
    raise ValueError("No data found. Run preprocessing first.")

print(f"📊 {len(df):,} samples, {df['label'].mean():.1%} spam")

# Receive and preprocess features
X_text = df['text'].fillna('')
y = df['label']

# Feature preparation - Essential feature 1: Combined TF-IDF + engineered features
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(X_text)

# Extract spam features for each text
spam_features = [extract_features(str(text)) + [len(str(text)), len(str(text).split())] 
                for text in X_text]
X_features = np.array(spam_features)
print(f"Extracted engineered features shape: {X_features.shape}")

# Scale the engineered features only (TF-IDF is already normalized)
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)

# Combine TF-IDF with scaled engineered features
X = hstack([X_tfidf, csr_matrix(X_features_scaled)])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model - Essential feature 2: Optimized LogisticRegression
model = LogisticRegression(C=10, random_state=42, max_iter=20000)
model.fit(X_train, y_train)

# Debug: Print training feature dimensions
print(f"Training features shape: {X_train.shape}")
print(f"TF-IDF features: {X_tfidf.shape[1]}")
print(f"Engineered features: {X_features.shape[1]}")
print(f"Expected total features: {X_tfidf.shape[1] + X_features.shape[1]}")

# Test feature extraction consistency
test_text = "sample test text"
test_spam_features = extract_features(test_text)
test_text_stats = [len(test_text), len(test_text.split())]
print(f"Sample feature extraction length: {len(test_spam_features + test_text_stats)}")

# Evaluation
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"🎯 AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Save model, vectorizer, and scaler
Path("saved_models").mkdir(exist_ok=True)
joblib.dump({
    'model': model, 
    'vectorizer': vectorizer, 
    'scaler': scaler
}, 'saved_models/model.joblib')
print("💾 Model saved")

# Test predictions - Essential feature 3: Real-time prediction capability
def predict_email(text):
    clean_text_input = clean_text(text)
    if not clean_text_input:
        return "Error: Empty text"
    
    try:
        # Transform text with TF-IDF
        X_tfidf = vectorizer.transform([clean_text_input]) 
        
        # Extract and scale engineered features
        spam_features = extract_features(clean_text_input)
        text_stats = [len(clean_text_input), len(clean_text_input.split())]
        X_features = np.array([spam_features + text_stats])
        
        # Debug: Check feature dimensions
        print(f"Debug - Spam features length: {len(spam_features)}")
        print(f"Debug - X_features shape: {X_features.shape}")
        
        # Scale the features using the fitted scaler
        X_features_scaled = scaler.transform(X_features)
        
        # Combine features - ensure both are sparse matrices
        X_combined = hstack([X_tfidf, csr_matrix(X_features_scaled)])
        X_combined = X_combined.tocsr()  # Ensure the matrix is in CSR format
        
        # Debug: Check final shape
        print(f"Debug - X_combined shape: {X_combined.shape}")
        
        probability = model.predict_proba(X_combined)[0, 1]
        prediction = "Spam" if probability > 0.5 else "Ham"
        return f"{prediction} ({probability:.3f})"
        
    except Exception as e:
        return f"Prediction error: {str(e)} - Feature shape mismatch likely"

# Test with sample emails
test_emails = [
    "FREE MONEY! Click now!",
    "Meeting at 2pm tomorrow",
    "Urgent: verify account",
    "Project update attached"
]

print("\n🧪 Test predictions:")
for i, email in enumerate(test_emails, 1):
    result = predict_email(email)
    print(f"{i}. {result} - {email[:30]}...")

print(f"\n🎉 Completed! AUC: {auc:.4f}")