# Logistic Regression Spam Classification
# Author: Your Name
# Date: 2025-09-30


import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from preprocessing import load_dataset, clean_text, extract_features

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

def evaluate_model(y_true, y_pred, y_prob):
    print("Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label=1))
    print("Recall:", recall_score(y_true, y_pred, pos_label=1))
    print("F1 Score:", f1_score(y_true, y_pred, pos_label=1))
    print("ROC AUC Score:", roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))

# Test predictions
def predict_email(tfidf, clf, text):
    clean_text_input = clean_text(text)
    if not clean_text_input:
        return "Error: Empty text"
    
    X = tfidf.transform([clean_text_input])
    y_prob = clf.predict_proba(X)[0, 1]
    label = 'Spam' if y_prob >= 0.5 else 'Ham'
    return label, y_prob

def main():
    # Load dataset
    df = load_dataset()
    if df is None:
        raise ValueError("No data found. Run preprocessing first.")
    print(f"📊 {len(df):,} samples, {df['label'].mean():.1%} spam")

    # Receive and preprocess features
    X_text = df['text'].fillna('')
    y = df['label']

    # Text vectorization using TfidfVectorizer 
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(X_text)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model - Essential feature 2: Optimized LogisticRegression
    clf = LogisticRegression(C=10, random_state=42, max_iter=20000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluation
    evaluate_model(y_test, y_pred, clf.predict_proba(X_test)[:, 1])

    # Feature importance analysis
    features_name = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    spam_words = coefs.argsort()[::-1]
    ham_words = coefs.argsort()
    print("Top 5 positive coefficients (spam indicators):")
    for coef in spam_words[:5]:
        print(f"{features_name[coef]:<20}: {coef:.2f}")
    print("Top 5 negative coefficients (ham indicators):")
    for coef in ham_words[:5]:
        print(f"{features_name[coef]:<20}: {coef:.2f}")

    # Save model and vectorizer
    joblib.dump(clf, MODEL_DIR / 'clf_model.joblib')
    joblib.dump(tfidf, MODEL_DIR / 'clf_vectorizer.joblib')
    print("💾 Logistic Regression Model saved")

    # Example usage
    test_emails = [
        "FREE MONEY! Click here to win $10000 now! Limited time offer!",
        "Hi John, let's meet for lunch tomorrow at 12pm. Looking forward to seeing you!",
        "Dear user, you have won a prize! claim now verification required",
        "Meeting agenda attached for project update discussion", 
        "Urgent: verify your bank account immediately or account will be closed",
        "Lunch plans? Let me know your preference for the restaurant"
    ]

    for example_email in test_emails:
        label, prob = predict_email(tfidf, clf, example_email)
        print(f"Prediction: {label} (Probability: {prob:.2f}) - {example_email:.30}")

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Logistic Regression Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    main()