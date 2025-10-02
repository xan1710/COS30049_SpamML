from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample documents
documents = [
    "I love coding.",
    "Coding is fun.",
    "I love learning new things."
]

# Using CountVectorizer
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)
print("Count Vectorizer Result:\n", count_matrix.toarray()) # type: ignore

# Using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("TF-IDF Vectorizer Result:\n", tfidf_matrix.toarray()) # type: ignore
