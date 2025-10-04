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
from scipy.sparse import hstack
import joblib
from preprocessing import load_cleaned_data, clean_text, extract_features

def prepare_features(texts, vectorizer=None, fit=False):
    """Prepare TF-IDF and engineered features"""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    
    X_tfidf = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    
    # Extract spam features for each text
    if isinstance(texts, pd.Series):
        spam_features = [extract_features(str(text)) + [len(str(text)), len(str(text).split())] 
                        for text in texts.fillna('')]
    else:
        text = texts[0] if isinstance(texts, list) else str(texts)
        spam_features = [extract_features(text) + [len(text), len(text.split())]]
    
    X_features = np.array(spam_features)

    # Combine TF-IDF and engineered features
    return hstack([X_tfidf, X_features]), vectorizer

def train_model(df):
    """Train logistic regression model"""
    # Prepare features
    X, vectorizer = prepare_features(df['text'], fit=True)
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = LogisticRegression(C=10, random_state=42, max_iter=20000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"🎯 AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    return model, vectorizer, auc

def predict_email(text, model, vectorizer):
    """Predict single email"""
    clean_text_input = clean_text(text)
    if not clean_text_input:
        return "Error: Empty text"
    
    X, _ = prepare_features([clean_text_input], vectorizer)
    probability = model.predict_proba(X)[0, 1]
    prediction = "Spam" if probability > 0.5 else "Ham"
    
    return f"{prediction} ({probability:.2f})"

def save_model(model, vectorizer, filename="model.joblib"):
    """Save model and vectorizer"""
    path = Path("saved_models") / filename
    path.parent.mkdir(exist_ok=True)
    joblib.dump({'model': model, 'vectorizer': vectorizer}, path)
    print(f"💾 Saved: {path}")

def load_model(filename="model.joblib"):
    """Load model and vectorizer"""
    path = Path("saved_models") / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    artifacts = joblib.load(path)
    print(f"📥 Loaded: {path}")
    return artifacts['model'], artifacts['vectorizer']

def main():
    """Main execution"""
    print("🚀 SPAM CLASSIFIER")
    
    # Load and prepare data
    df = load_cleaned_data()
    if df is None:
        raise ValueError("No data found. Run preprocessing first.")
    
    print(f"📊 {len(df):,} samples, {df['label'].mean():.1%} spam")
    
    # Train model
    model, vectorizer, auc = train_model(df)
    
    # Save model
    save_model(model, vectorizer)
    
    # Test predictions
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
        result = predict_email(email, model, vectorizer)
        print(f"{i}. {result} - {email[:30]}...")
    
    print(f"\n🎉 Completed! AUC: {auc:.4f}")

if __name__ == "__main__":
    main()