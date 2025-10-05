import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import load_dataset

# Load the saved models
print("Loading models...")
log_reg_data = joblib.load('saved_models/log_reg_model.joblib')
knn_model = joblib.load('saved_models/knn_model.joblib')  # assuming this exists

log_reg_model = log_reg_data['model']
vectorizer = log_reg_data['vectorizer']

# Load test data
df = load_dataset()
if df is None:
    print("Error: No dataset found!")
    exit()

# Prepare test data
X_text = df['text'].fillna('')
y_true = df['label']

# Get predictions from both models
print("Getting predictions...")

# Logistic regression predictions
X_tfidf = vectorizer.transform(X_text)
log_reg_pred = log_reg_model.predict(X_tfidf)
log_reg_proba = log_reg_model.predict_proba(X_tfidf)[:, 1]

# KNN predictions (assuming numeric features)
feature_cols = ['number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']
X_numeric = df[feature_cols]
knn_pred = knn_model.predict(X_numeric)
knn_proba = knn_model.predict_proba(X_numeric)[:, 1]

# Calculate accuracies
log_reg_acc = accuracy_score(y_true, log_reg_pred)
knn_acc = accuracy_score(y_true, knn_pred)

print(f"Logistic Regression Accuracy: {log_reg_acc:.3f}")
print(f"KNN Accuracy: {knn_acc:.3f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy comparison
models = ['Logistic Regression', 'KNN']
accuracies = [log_reg_acc, knn_acc]

axes[0,0].bar(models, accuracies, color=['blue', 'orange'])
axes[0,0].set_title('Model Accuracy Comparison')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_ylim(0, 1)

# Add accuracy values on bars
for i, v in enumerate(accuracies):
    axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center')

# Confusion matrices
cm_log = confusion_matrix(y_true, log_reg_pred)
cm_knn = confusion_matrix(y_true, knn_pred)

sns.heatmap(cm_log, annot=True, fmt='d', ax=axes[0,1], cmap='Blues')
axes[0,1].set_title('Logistic Regression Confusion Matrix')

sns.heatmap(cm_knn, annot=True, fmt='d', ax=axes[1,0], cmap='Oranges')
axes[1,0].set_title('KNN Confusion Matrix')

# Probability distributions
axes[1,1].hist(log_reg_proba, bins=20, alpha=0.7, label='Log Reg', color='blue')
axes[1,1].hist(knn_proba, bins=20, alpha=0.7, label='KNN', color='orange')
axes[1,1].set_title('Prediction Probability Distribution')
axes[1,1].set_xlabel('Probability')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Print classification reports
print("\n=== Logistic Regression Report ===")
print(classification_report(y_true, log_reg_pred, target_names=['Ham', 'Spam']))

print("\n=== KNN Report ===")
print(classification_report(y_true, knn_pred, target_names=['Ham', 'Spam']))

# Determine winner
if log_reg_acc > knn_acc:
    winner = "Logistic Regression"
    diff = log_reg_acc - knn_acc
else:
    winner = "KNN"
    diff = knn_acc - log_reg_acc

print(f"\n🏆 Winner: {winner} (by {diff:.3f} accuracy points)")