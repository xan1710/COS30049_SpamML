# COS30049_SpamML

This project implements three machine learning models for spam email detection, Logistic Regression, (KNN), and K-Means Clustering.

## 1. Environment Setup

We recommend using [conda](https://docs.conda.io/en/latest/) for environment management.

# Fork the repository to your own workspace
git clone https://github.com/your-name-here/COS30049_SpamML.git

# Create a new conda environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate cos30049_ml

## 2. Preprocessing Data

Use the preprocessing script to clean and prepare the datasets:

python preprocessing.py

The input and output files are in the datasets/ folder.
Adjust the preprocessing.py as needed to include extra datasets.

## 3. Training Models

Train models using the provided scripts. Each model uses different features and approaches:

### Logistic Regression Model
Uses TF-IDF text vectorization with English stop words removal:
```bash
python log_reg.py
```
- **Features**: TF-IDF vectors ( with a max of 1000 features)
- **Output**: Classification with probability scores
- **Saves**: `clf_model.joblib` and `clf_vectorizer.joblib`

### K-Nearest Neighbors (KNN) Model
Uses engineered numerical features with customizable k-value:
```bash
# default k=5
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

## 4. Using the Models for Prediction

### Logistic Regression Predictions:
The logistic regression model includes built-in prediction examples and can handle raw text input using tf-idf:

### KNN and K-Means Predictions:
For KNN and K-Means models, numerical features are extracted first.
Each training script also includes example predictions with test emails.

## 5. Model Evaluation & Visualization
Each model provides comprehensive evaluation metrics and visualizations:

### Evaluation Metrics
All models output the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC AUC Score**: this is the under the receiver operating characteristic curve (Logistic Regression only)

### Visualizations Generated
- **Confusion Matrix**
- **Silhouette Analysis**

### Additional Analysis
- Use "model_dataset_analysis.py" for comparative analysis across datasets
- Visualizations are automatically saved in the "visualizations/" folder
- The K-Means model includes silhouette analysis for k-values from 2 to 10

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) [year] [fullname]

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
