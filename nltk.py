# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
# nltk.download('punkt')
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# # Sample data
# text_data = [
#     "This movie is amazing and I love it!",
#     "I hate this film, it's terrible.",
#     "A decent film, not great but okay.",
#     "Absolutely wonderful performance by the actors."
# ]
# labels = [1, 0, 1, 1] # 1 for positive, 0 for negative

# # 1. NLTK for preprocessing
# stop_words = set(stopwords.words('english'))
# processed_texts = []
# for text in text_data:
#     words = word_tokenize(text.lower())
#     filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
#     processed_texts.append(" ".join(filtered_words))

# # 2. Feature Extraction (e.g., TF-IDF)
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(processed_texts)
# y = labels

# # 3. Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Scikit-learn for Logistic Regression
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 5. Evaluation
# predictions = model.predict(X_test)
# print(classification_report(y_test, predictions))