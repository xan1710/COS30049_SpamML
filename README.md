# COS30049_SpamML

A machine learning project for spam email detection using various models (KNN, Logistic Regression, etc.). This guide will help you set up the environment, preprocess data, train models, and use them for prediction.

## 1. Environment Setup

We recommend using [conda](https://docs.conda.io/en/latest/) for environment management.

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-name-here/COS30049_SpamML.git
cd COS30049_SpamML

# Create a new conda environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate cos30049_ml

# (Optional) If you want to install manually:
# conda create -n cos30049_ml python=3.13
# conda activate cos30049_ml
# pip install -r requirements.txt
```

## 2. Preprocessing Data

Preprocessing scripts are provided to clean and prepare the datasets.

```bash
# Run preprocessing (example)
python preprocessing.py
```

- Input and output files are in the `datasets/` folder.
- Adjust the script as needed for your dataset.

## 3. Training Models

You can train different models using the provided scripts:

```bash
# Train KNN model
python knn.py

# Train Logistic Regression model
python log_reg.py
```

- Trained models are saved in the `saved_models/` directory.

## 4. Using Models for Prediction

You can use the trained models to make predictions on new data:

```python
# Example usage in Python
from joblib import load
import pandas as pd

# Load model and vectorizer
model = load('saved_models/clf_model.joblib')
vectorizer = load('saved_models/clf_vectorizer.joblib')

# Prepare your input data (as a DataFrame or Series)
X_new = vectorizer.transform(["Your email text here"])

# Predict
prediction = model.predict(X_new)
print('Spam' if prediction[0] == 1 else 'Not Spam')
```

Or use/modify the provided scripts for batch predictions.

## 5. Visualization & Analysis

- Visualizations are saved in the `visualizations/` folder.
- Use or modify `model_dataset_analysis.py` for further analysis.

## 6. Requirements

- Python 3.13
- See `requirements.txt` or `environment.yml` for all dependencies.

---

For any issues, please open an issue or contact the maintainer.
