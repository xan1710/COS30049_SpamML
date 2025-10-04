import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocessing import load_dataset

def main():
    # Load cleaned data. change here if using a different cleaned dataset
    # (pick from one of the individual cleaned datasets or the combined one).
    df = load_dataset('combined_email_dataset.csv')
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
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Evaluate. we print some useful metrics including the
    # models accuracy in classifying the test set. We also
    # print a confusion matrix.
    y_pred = knn.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    #print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

if __name__ == "__main__":
    main()