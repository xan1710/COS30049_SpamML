import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparsec
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from preprocessing import load_dataset

def main():
    # show help, let user choose which cleaned dataset to use & specify k val
    default_dataset = 'combined_email_dataset.csv'
    default_dataset_help = 'name of the cleaned dataset file in the cleaned_datasets folder'
    parser = argparse.ArgumentParser(description="the knn model trainer. pick a cleaned dataset to train the model with.")
    parser.add_argument('--dataset', type=str, default=default_dataset, help=default_dataset_help)
    parser.add_argument('--k', type=int, default=5, help='specify k value')
    args = parser.parse_args()

    # print the name of cleaned dataset & k value being used to train
    print(f"Using cleaned dataset: {args.dataset}")
    print(f"Using k value: {args.k}")

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

    # Evaluate. we print some useful metrics including the
    # models accuracy in classifying the test set. We also
    # print a confusion matrix.
    y_pred = knn.predict(X_test_scaled)

    # Save the model
    model_dir = Path("saved_models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(knn, model_dir / "knn_model.joblib")
    joblib.dump(scaler, model_dir / "knn_scaler.joblib")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # plot a confusion matrix with test results
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('KNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    main()