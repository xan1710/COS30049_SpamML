# COS30049_SpamML

This project implements three machine learning models for spam email detection: **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **K-Means Clustering**. Each model uses different approaches and features to classify emails as spam or ham (legitimate emails). Follow this guide to set up the environment, preprocess data, train models, and use them for prediction.

## 1. Environment Setup

We recommend using [conda](https://docs.conda.io/en/latest/) for environment management.

```bash
# Fork the repository to your own workspace
git clone https://github.com/your-name-here/COS30049_SpamML.git
cd COS30049_SpamML

# Create a new conda environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate cos30049_ml

# (Optional) Manual installation:
# conda create -n cos30049_ml python=3.13
# conda activate cos30049_ml
# pip install -r requirements.txt
```

## 2. Preprocessing Data


Use the preprocessing script to clean and prepare the datasets:

```bash
python preprocessing.py
```

- Input and output files are in the `datasets/` folder.
- Adjust the script as needed for your dataset.

## 3. Training Models

Train models using the provided scripts. Each model uses different features and approaches:

### Logistic Regression Model
Uses TF-IDF text vectorization with English stop words removal:
```bash
python log_reg.py
```
- **Features**: TF-IDF vectors (max 1000 features)
- **Output**: Classification with probability scores
- **Saves**: `clf_model.joblib` and `clf_vectorizer.joblib`

### K-Nearest Neighbors (KNN) Model
Uses engineered numerical features with customizable k-value:
```bash
# Default k=5
python knn.py

# Custom dataset and k-value
python knn.py --dataset cleaned_emails.csv --k 7
```
- **Features**: `number_ratio`, `special_char_ratio`, `spam_words`, `text_length`, `word_count`
- **Preprocessing**: StandardScaler normalization
- **Saves**: `knn_model.joblib` and `knn_scaler.joblib`

### K-Means Clustering Model
Unsupervised clustering with silhouette analysis:
```bash
python kmeans.py
```
- **Features**: Same numerical features as KNN
- **Analysis**: Silhouette scores for k=2 to k=10
- **Saves**: `kmeans_model.joblib` and `kmeans_scaler.joblib`

All trained models are saved in the `saved_models/` directory with corresponding scalers/vectorizers.

## 4. Using Models for Prediction

### Logistic Regression Predictions
The logistic regression model includes built-in prediction examples and can handle raw text input:

```python
from joblib import load
from preprocessing import clean_text

# Load model and vectorizer
model = load('saved_models/clf_model.joblib')
vectorizer = load('saved_models/clf_vectorizer.joblib')

# Predict with probability
def predict_email(tfidf, clf, text):
    clean_text_input = clean_text(text)
    X = tfidf.transform([clean_text_input])
    y_prob = clf.predict_proba(X)[0, 1]
    label = 'Spam' if y_prob >= 0.5 else 'Ham'
    return label, y_prob

# Example usage
text = "FREE MONEY! Click here to win $10000 now!"
label, probability = predict_email(vectorizer, model, text)
print(f"Prediction: {label} (Probability: {probability:.2f})")
```

### KNN and K-Means Predictions
For KNN and K-Means models, you need to extract numerical features first:

```python
from joblib import load
import pandas as pd
from preprocessing import extract_features

# Load KNN model and scaler
knn_model = load('saved_models/knn_model.joblib')
knn_scaler = load('saved_models/knn_scaler.joblib')

# Prepare features for a new email
email_text = "Your email text here"
features = extract_features(pd.Series([email_text]))
features_scaled = knn_scaler.transform([features.iloc[0]])

# Predict
prediction = knn_model.predict(features_scaled)
print('Spam' if prediction[0] == 1 else 'Ham')
```

Each training script also includes example predictions with test emails.

## 5. Model Evaluation & Visualization

Each model provides comprehensive evaluation metrics and visualizations:

### Evaluation Metrics
All models output the following metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for spam detection
- **Recall**: Sensitivity for spam detection  
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC Score**: Area under the receiver operating characteristic curve (Logistic Regression only)

### Visualizations Generated
- **Confusion Matrix**: Visual representation of classification results
  - `logreg_confusion_matrix.png` - Logistic Regression results
  - `knn_confusion_matrix.png` - KNN results
- **Silhouette Analysis**: K-Means clustering quality analysis
  - `kmeans_silhouette_analysis.png` - Optimal cluster analysis

### Additional Analysis
- Use `model_dataset_analysis.py` for comparative analysis across datasets
- Visualizations are automatically saved in the `visualizations/` folder
- The K-Means model includes silhouette analysis for k-values from 2 to 10

## 6. Requirements

- Python 3.13
- See `requirements.txt` or `environment.yml` for all dependencies.

---

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 COS30049 Spam Email Detection Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
**Important:** If you plan to use this project or its datasets for commercial purposes, please ensure you have the appropriate licenses and permissions from the original dataset providers.

---