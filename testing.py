import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris # Example dataset

load_iris()

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

df = pd.read_csv('emails.csv')  # Assuming a CSV file with 'text' and 'label' columns
print(df.head())
print(df.info())

df = pd.read_csv('hf://datasets/bourigue/data_email_spam/spam_email_dataset.csv')
print(df.head())
print(df.info())
# Note: This is a simplified example of using Logistic Regression with sklearn.
# The actual logistic_regression_pipeline.py and preprocessing.py files contain more complex logic and features.
# The changes made were to remove print statements that output total features and subject extraction logic.
# The following code snippets are from the files logistic_regression_pipeline.py and preprocessing.py
# that were recently edited to remove certain print statements and subject extraction logic.
# The changes are indicated with --- IGNORE --- comments.
# Please refer to the original files for complete implementations.
# The changes made are as follows:
# In preprocessing.py, removed the extraction of 'subject' from the text data.
# In logistic_regression_pipeline.py, removed the print statement that outputs total features.
# The context of these changes is to streamline the output and focus on essential information.
# The following code snippets are from the files logistic_regression_pipeline.py and preprocessing.py
# that were recently edited to remove certain print statements and subject extraction logic.
