import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from preprocessing import DATA_DIR

# Load your dataset
df = pd.read_csv(DATA_DIR / "cleaned_datasets" / "combined_email_dataset.csv")

# Basic preprocessing

X_text = df['text']
y = df['label']

# Text vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# --- Classification ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

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

# Save the classifier and vectorizer
joblib.dump(clf, 'spam_classifier.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

# --- Clustering ---
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# Evaluation for clustering
score = silhouette_score(X, clusters)
print("Clustering Silhouette Score:", score)

# Visualize cluster sizes
plt.bar([0, 1], [(clusters == 0).sum(), (clusters == 1).sum()])
plt.title('Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Number of Samples')
plt.show()

# Save the clustering model
joblib.dump(kmeans, 'spam_kmeans.joblib')