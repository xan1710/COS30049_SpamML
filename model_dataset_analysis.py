import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import load_dataset

warnings.filterwarnings('ignore')

# Load the dataset and models
df = pd.read_csv('datasets/cleaned_datasets/combined_email_dataset.csv')
clf_model = joblib.load('saved_models/clf_model.joblib')
clf_vectorizer = joblib.load('saved_models/clf_vectorizer.joblib')
knn_model = joblib.load('saved_models/knn_model.joblib')
knn_scaler = joblib.load('saved_models/knn_scaler.joblib')


# Create the combined visualization
fig = plt.figure(figsize=(10, 8))
fig.legend(["Logistic Regression", "KNN"], loc='upper center', ncol=2, fontsize=12)

# Prepare test data
X_text = df['text'].fillna('')
y_true = df['label']

# Logistic regression predictions
X_tfidf = clf_vectorizer.transform(X_text)
log_reg_pred = clf_model.predict(X_tfidf)
log_reg_proba = clf_model.predict_proba(X_tfidf)[:, 1]

# KNN predictions
feature_cols = ['number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']
X_numeric = df[feature_cols]
X_numeric_scaled = knn_scaler.transform(X_numeric)
knn_pred = knn_model.predict(X_numeric_scaled)
knn_proba = knn_model.predict_proba(X_numeric_scaled)[:, 1]

# Calculate accuracies
log_reg_acc = accuracy_score(y_true, log_reg_pred)
knn_acc = accuracy_score(y_true, knn_pred)

### MODEL COMPARISON PLOTS ###

# 1. Model Accuracy Comparison
plt.subplot(2, 2, 1)
models = ['Logistic Regression', 'KNN']
accuracies = np.array([log_reg_acc, knn_acc])
bars = plt.bar(models, accuracies, color=["#0099ff", "#ff7700"])
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add accuracy values on bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# 2. Confusion Matrices Comparison
plt.subplot(2, 2, 2)
cm_log = confusion_matrix(y_true, log_reg_pred)
cm_knn = confusion_matrix(y_true, knn_pred)

# Create side-by-side confusion matrices
x_pos = [0, 1, 2, 3]  # positions for the bars
cm_values = [cm_log[0,0], cm_log[1,1], cm_knn[0,0], cm_knn[1,1]]  # TP, TN values
colors = ['#0099ff', '#0099ff', '#ff7700', '#ff7700']
labels = ['LogReg\nTN', 'LogReg\nTP', 'KNN\nTN', 'KNN\nTP']

bars = plt.bar(x_pos, cm_values, color=colors, alpha=0.8)
plt.title('True Predictions Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.xticks(x_pos, labels)

# Add values on bars
for i, v in enumerate(cm_values):
    plt.text(x_pos[i], v + max(cm_values)*0.01, str(v), ha='center', fontweight='bold')

### DATASET VISUALIZATION PLOTS ###

# 3. Class Distribution
plt.subplot(2, 2, 3)
class_counts = df['label'].value_counts()
plt.pie(class_counts.to_numpy(), labels=['Ham', 'Spam'], autopct='%.2f%%')
plt.title('Email Classification Distribution', fontsize=14, fontweight='bold')

# 4. Feature Correlation Matrix
plt.subplot(2, 2, 4)
numeric_cols = ['label', 'number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0, fmt='.2f', square=True)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

plt.suptitle('Email Spam Detection - Model Comparison & Dataset Analysis', fontsize=18, fontweight='bold', y=0.98)
plt.show()

# Print comprehensive results summary. Display dataset overview 
# and classification reports for both models 
print("DATASET OVERVIEW:")
print(f"Total emails: {len(df)}")
print(f"Ham emails: {len(df[df['label'] == 0])}")
print(f"Spam emails: {len(df[df['label'] == 1])}")

print("===== Logistic Regression Report =====")
print(classification_report(y_true, log_reg_pred, target_names=['Ham', 'Spam']))

print("===== KNN Report =====")
print(classification_report(y_true, knn_pred, target_names=['Ham', 'Spam']))