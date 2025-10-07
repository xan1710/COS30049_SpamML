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

# Evaluation function containing different metrics for evaluating the performance 
# of a model on a test set. This includes accuracy, precision, recall, f1-score and roc-auc score
def evaluate_model(y_true, y_pred, y_prob):
    print("Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label=1))
    print("Recall:", recall_score(y_true, y_pred, pos_label=1))
    print("F1 Score:", f1_score(y_true, y_pred, pos_label=1))
    print("ROC AUC Score:", roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))

# Test predictions of the model from text message and probability
def predict_email(tfidf, clf, text):
    clean_text_input = clean_text(text)
    if not clean_text_input:
        return "Error: Empty text"
    
    X = tfidf.transform([clean_text_input])
    y_prob = clf.predict_proba(X)[0, 1]
    label = 'Spam' if y_prob >= 0.5 else 'Ham'
    return label, y_prob

def main():
    # Load dataset from datasets folder
    df = load_dataset()
    if df is None:
        raise ValueError("No data found. Run preprocessing first.")

    # From the loaded dataset, we extract text and labels for X and y
    # .fillna('') is used to handle any missing values in the text column
    X_text = df['text'].fillna('')
    y = df['label']
    
    # Text vectorization using TfidfVectorizer with parameters 
    # to limit features and remove stop words 
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X = tfidf.fit_transform(X_text)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and train a LogisticRegression classifier with parameters
    # to prevent overfitting. 
    clf = LogisticRegression(C=1.0, random_state=42, solver='liblinear', max_iter=20000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calling evaluation method to print metrics for the model 
    # on the test set. We also pass predicted probabilities
    # for ROC AUC calculation.
    evaluate_model(y_test, y_pred, clf.predict_proba(X_test)[:, 1])
    
    # Save model and vectorizer
    joblib.dump(clf, MODEL_DIR / 'clf_model.joblib')
    joblib.dump(tfidf, MODEL_DIR / 'clf_vectorizer.joblib')
    print("Logistic Regression Model saved")

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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Logistic Regression Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    main()