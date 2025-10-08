# kmeans.py
# Author: PixelNest Labs
# Date: 2025-10-08
# This module contains functions for training and evaluating a K-means clustering model
# for spam detection with silhouette analysis.

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from preprocessing import load_dataset

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

# Evaluation function containing different metrics for evaluating the performance 
# of a clustering model. This includes accuracy, precision, recall, f1-score
def evaluate_model(y_true, y_pred):
    print("Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label=1))
    print("Recall:", recall_score(y_true, y_pred, pos_label=1))
    print("F1 Score:", f1_score(y_true, y_pred, pos_label=1))

def main():
    # Load cleaned data from the combined dataset
    df = load_dataset('combined_email_dataset.csv')
    if df is None:
        raise ValueError("No cleaned datasets found. Run preprocessing first.")

    # Set the numeric feature columns and the labels as X and y
    feature_columns = ['number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']
    X = df[feature_columns]
    y = df['label']

    # Scale the features (important for K-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the K-means model with 2 clusters (Ham and Spam)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    # Map clusters to actual labels (0=Ham, 1=Spam)
    # We need to determine which cluster corresponds to which class
    cluster_0_spam_ratio = np.mean(y[cluster_labels == 0])
    cluster_1_spam_ratio = np.mean(y[cluster_labels == 1])
    
    # If cluster 0 has more spam, map it to 1, otherwise keep as is
    if cluster_0_spam_ratio > cluster_1_spam_ratio:
        y_pred = 1 - cluster_labels  # Flip the labels
    else:
        y_pred = cluster_labels

    # Save the model and scaler
    joblib.dump(kmeans, MODEL_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "kmeans_scaler.joblib")
    
    # Evaluate the clustering results
    evaluate_model(y, y_pred)

    # Silhouette analysis for different k values (2 to 10)
    cluster_range = range(2, 11)
    silhouette_scores = []
    
    # Calculate silhouette scores for each k value and store them
    for n_clusters in cluster_range:
        kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        temp_labels = kmeans_temp.fit_predict(X_scaled)
        sil_avg = silhouette_score(X_scaled, temp_labels)
        silhouette_scores.append(sil_avg)
    
    # Plot silhouette analysis
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=2, color='red', linestyle='--', alpha=0.7, label=f'Selected k=2 (Score: {sil_score:.3f})')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for K-means Clustering')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()