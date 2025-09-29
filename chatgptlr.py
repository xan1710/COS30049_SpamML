# spam_lr_pipeline.py
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import joblib

# ---------- 1) text cleaning (keeps capitalization for numeric features)
def clean_text_keep_case(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    # remove script tags and HTML tags quickly (regex - safe for many emails)
    s = re.sub(r'(?is)<script.*?>.*?</script>', ' ', s)
    s = re.sub(r'<[^>]+>', ' ', s)
    # replace urls/emails/numbers with tokens (helpful features)
    s = re.sub(r'http[s]?://\S+|www\.\S+', ' URL ', s)
    s = re.sub(r'\S+@\S+', ' EMAIL ', s)
    s = re.sub(r'\d+', ' NUM ', s)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'[^A-Za-z0-9\s\@\.\!\?]', ' ', s)  # remove weird chars
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

# ---------- 2) numeric feature extractor (returns sparse matrix)
def extract_numeric_features(texts):
    stopwords = ENGLISH_STOP_WORDS
    feats = []
    for s in texts:
        if s is None:
            s = ""
        s = str(s)
        char_count = len(s)
        words = s.split()
        word_count = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0.0
        url_count = len(re.findall(r'URL', s))  # we replaced urls with 'URL'
        email_count = len(re.findall(r'EMAIL', s))
        exclaim_count = s.count('!')
        question_count = s.count('?')
        digit_count = sum(ch.isdigit() for ch in s)
        uppercase_words = sum(1 for w in words if any(c.isupper() for c in w))
        unique_word_count = len(set(words))
        stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / word_count if word_count > 0 else 0.0

        feats.append([
            char_count, word_count, avg_word_len, url_count, email_count,
            exclaim_count, question_count, digit_count, uppercase_words,
            unique_word_count, stopword_ratio
        ])
    X = np.array(feats, dtype=float)
    # convert numeric matrix to sparse so ColumnTransformer results stay sparse
    return sparse.csr_matrix(X)

# ---------- 3) Build Pipeline
def build_pipeline():
    # TF-IDF settings tuned for a fast but strong baseline
    tf_word = TfidfVectorizer(preprocessor=clean_text_keep_case,
                              analyzer='word',
                              ngram_range=(1,2),
                              max_features=10000,
                              min_df=3,
                              sublinear_tf=True)

    tf_char = TfidfVectorizer(preprocessor=clean_text_keep_case,
                              analyzer='char_wb',
                              ngram_range=(3,5),
                              max_features=3000,
                              min_df=3,
                              sublinear_tf=True)

    column_transformer = ColumnTransformer(
        transformers=[
            ('tf_word', tf_word, 'text'),
            ('tf_char', tf_char, 'text'),
            ('num', FunctionTransformer(extract_numeric_features, validate=False), 'text'),
        ],
        sparse_threshold=0.3  # keep sparse if possible
    )

    clf = LogisticRegression(solver='saga', class_weight='balanced', max_iter=20000, random_state=42)

    pipeline = Pipeline([
        ('features', column_transformer),
        ('clf', clf)
    ])

    return pipeline

# ---------- 4) Example train / tune / evaluate
def train_and_evaluate(df):
    # normalize label to 0/1
    if df['label'].dtype == object:
        df['label'] = df['label'].map(lambda x: 1 if str(x).lower().startswith('s') else 0)

    X = df[['text']]  # DataFrame (ColumnTransformer expects column names)
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)

    pipeline = build_pipeline()

    # small grid for fast tuning; expand if you have time
    param_grid = {
        'clf__C': [0.1, 1.0, 5.0],
        'clf__penalty': ['l2']  # keep L2 by default (fast & robust)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best = grid.best_estimator_

    # predictions & metrics
    y_pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]

    print("\nClassification report (at threshold=0.5):\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass

    # Threshold tuning (optimize F1 on test set or a validation split)
    prec, rec, thresh = precision_recall_curve(y_test, proba)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_index = np.nanargmax(f1_scores)
    best_thresh = thresh[best_index] if best_index < len(thresh) else 0.5
    print(f"Best threshold (max F1 on test set): {best_thresh:.3f}  - F1={f1_scores[best_index]:.3f}")

    # Save model
    joblib.dump(best, 'spam_lr_pipeline.joblib')
    print("Model saved to spam_lr_pipeline.joblib")

    return best

# ---------- 5) Quick EDA helpers (optional)
def quick_eda(df):
    print("Label distribution:\n", df['label'].value_counts(normalize=True))
    df['len'] = df['text'].astype(str).map(len)
    print("\nText length describe:\n", df['len'].describe())
    # show a few spam/ham examples to inspect
    print("\nSample spam:\n", df[df['label'] == 1]['text'].sample(3).tolist())
    print("\nSample ham:\n", df[df['label'] == 0]['text'].sample(3).tolist())

# ---------- usage
if __name__ == '__main__':
    # Replace this with your file/loading logic
    df = pd.read_csv('emails.csv')  # must contain 'text' and 'label'
    quick_eda(df)
    model = train_and_evaluate(df)
