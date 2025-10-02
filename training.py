
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from preprocessing import DATA_DIR
# Load your dataset
df = pd.read_csv(DATA_DIR / "cleaned_datasets" / "combined_email_dataset.csv")
# df = pd.read_csv('cleaned_emails.csv')
# Basic preprocessing
X_text = df['text']
y = df['label']
# y = df['spam']
# Text vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Evaluation
print("Classification Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label=1))
print("Recall:", recall_score(y_test, y_pred, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred, pos_label=1))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model and vectorizer
joblib.dump(model, 'logistic_regression_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')