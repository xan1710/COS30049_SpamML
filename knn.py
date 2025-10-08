# knn.py
# Author: PixelNest Labs
# Date: 2025-10-05
# This module contains functions for training and evaluating a KNN model
# for spam detection.

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from preprocessing import load_dataset

# Evaluation function containing different metrics for evaluating the performance 
# of a model on both train and test sets. This includes accuracy, precision, recall, f1-score and roc-auc score
def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
    print("Evaluation Metrics:")
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred, pos_label=1))
    print("Recall:", recall_score(y_test, y_test_pred, pos_label=1))
    print("F1 Score:", f1_score(y_test, y_test_pred, pos_label=1))
    print("ROC AUC Score:", roc_auc_score(y_test, y_test_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Ham', 'Spam']))

def main():
    # show help, let user choose which cleaned dataset to use & specify k val
    default_pop_up_msg = "the knn model trainer. pick a cleaned dataset to train the model with."
    default_dataset = 'combined_email_dataset.csv'
    default_dataset_help = 'name of the cleaned dataset file in the cleaned_datasets folder'
    parser = argparse.ArgumentParser(description=default_pop_up_msg)
    parser.add_argument('--dataset', type=str, default=default_dataset, help=default_dataset_help)
    parser.add_argument('--k', type=int, default=5, help='specify k value')
    args = parser.parse_args()

    # print the name of cleaned dataset & k value being used to train
    print(f"Using cleaned dataset: {args.dataset}")
    print(f"Using k value: {args.k}\n")

    # Load cleaned data. change here if using a different cleaned dataset
    # (pick from one of the individual cleaned datasets or the combined one).
    df = load_dataset(args.dataset)
    if df is None:
        raise ValueError("No cleaned datasets found. Run preprocessing first.")

    # set the numeric feature columns and the labels as X and y
    feature_columns = ['number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']
    X = df[feature_columns]
    y = df['label']

    # we split the dataset into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # we scale the features. this is important for knn.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # we train the knn model. 'n_neigbors' is the 'k' value.
    # play around with 'k' val to see how it affects results!
    knn = KNeighborsClassifier(n_neighbors=args.k)
    knn.fit(X_train_scaled, y_train)

    # Get predictions for both train and test sets
    y_train_pred = knn.predict(X_train_scaled)
    y_test_pred = knn.predict(X_test_scaled)

    # Save the model
    model_dir = Path("saved_models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(knn, model_dir / "knn_model.joblib")
    joblib.dump(scaler, model_dir / "knn_scaler.joblib")
    
    # Evaluate the model on both train and test sets
    evaluate_model(y_train, y_train_pred, y_test, y_test_pred)

    # plot a confusion matrix with test results
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:\n", cm)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('KNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    main()
