# Simplified k-Nearest Neighbors Spam Classification Pipeline
# Author: Your Name
# Date: 2025-10-02

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
from preprocessing import load_dataset, clean_text, extract_features

# Load dataset
df = load_dataset()
if df is None:
    raise ValueError("No data found. Run preprocessing first.")

print(f"📊 {len(df):,} samples, {df['label'].mean():.1%} spam")

# Feature preparation - Essential feature 1: Combined TF-IDF + engineered features
vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df['text'].fillna(''))

# Extract spam features for each text
spam_features = [extract_features(str(text)) + [len(str(text)), len(str(text).split())] 
                for text in df['text'].fillna('')]
X_features = np.array(spam_features)
X = hstack([X_tfidf, X_features])
y = df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # type: ignore

# Train model - Essential feature 2: Optimized k-Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) # type: ignore

print(f"🎯 AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Save model and vectorizer
Path("saved_models").mkdir(exist_ok=True)
joblib.dump({'model': model, 'vectorizer': vectorizer}, 'saved_models/knn_model.joblib')
print("💾 kNN Model saved")

# Test predictions - Essential feature 3: Real-time prediction capability
def predict_email(text):
    clean_text_input = clean_text(text)
    if not clean_text_input:
        return "Error: Empty text"
    
    X_tfidf = vectorizer.transform([clean_text_input])
    spam_features = extract_features(clean_text_input)
    text_stats = [len(clean_text_input), len(clean_text_input.split())]
    X_features = np.array([spam_features + text_stats])
    X_combined = hstack([X_tfidf, X_features]).tocsr()
    
    probability = model.predict_proba(X_combined)[0][1] # type: ignore
    prediction = "Spam" if probability > 0.5 else "Ham"
    return f"{prediction} ({probability:.3f})"

# Test with sample emails
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

print("\n🧪 Test predictions:")
for i, email in enumerate(test_emails, 1):
    result = predict_email(email)
    print(f"{i}. {result} - {email[:30]}...")

print(f"\n🎉 Completed! AUC: {auc:.4f}")