# Simplified Logistic Regression Spam Classification Pipeline
# Author: Your Name
# Date: 2025-09-30

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
from preprocessing import load_cleaned_data, clean_text, extract_spam_features

MODELS_DIR = Path("saved_models")

class SpamClassifier:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            stop_words='english', max_features=max_features, 
            ngram_range=(1, 2), min_df=2, max_df=0.95
        )
        self.model = LogisticRegression(C=10, random_state=42, max_iter=20000)
        self.feature_names = []
    
    def prepare_features(self, df):
        """Extract and combine TF-IDF and engineered features"""
        X_tfidf = self.vectorizer.fit_transform(df['text'].fillna(''))
        
        # Get numeric feature columns (exclude text and label)
        self.feature_names = [col for col in df.columns if col not in ['text', 'label']]
        X_features = df[self.feature_names].fillna(0).values
        
        X = hstack([X_tfidf, X_features])
        return X, df['label'].values
    
    def train(self, X, y):
        """Train the model and return evaluation metrics"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"🎯 Test Accuracy: {results['accuracy']:.4f}")
        print(f"🎯 Test AUC: {results['auc']:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        return results
    
    def predict(self, text):
        """Predict spam probability for a single email"""
        clean_text_input = clean_text(text)
        if not clean_text_input:
            return {'error': 'Empty text after cleaning'}
        
        # Prepare features
        X_tfidf = self.vectorizer.transform([clean_text_input])
        spam_features = extract_spam_features(clean_text_input)
        text_stats = [len(clean_text_input), len(clean_text_input.split())]
        X_features = np.array([spam_features + text_stats])
        X_combined = hstack([X_tfidf, X_features])
        
        # Predict
        prediction = self.model.predict(X_combined)[0] # type: ignore
        probability = self.model.predict_proba(X_combined)[0, 1] #
        
        return {
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'spam_probability': float(probability),
            'confidence': float(max(self.model.predict_proba(X_combined)[0]))
        }
    
    def save(self, filename="spam_classifier.joblib"):
        """Save model artifacts"""
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / filename
        
        artifacts = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names
        }
        
        joblib.dump(artifacts, model_path)
        print(f"💾 Model saved to: {model_path}")
        return model_path
    
    @classmethod
    def load(cls, filename="spam_classifier.joblib"):
        """Load saved model"""
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        artifacts = joblib.load(model_path)
        classifier = cls()
        classifier.model = artifacts['model']
        classifier.vectorizer = artifacts['vectorizer']
        classifier.feature_names = artifacts['feature_names']
        
        print(f"📥 Model loaded from: {model_path}")
        return classifier

def test_predictions(classifier):
    """Test model with sample emails"""
    test_emails = [
        "FREE MONEY! Click here to win $10000 now!",
        "Hi John, let's meet for lunch tomorrow at 12pm.",
        "Urgent: verify your bank account immediately",
        "Meeting agenda attached for project update"
    ]
    
    print("\n🧪 Testing predictions:")
    for i, email in enumerate(test_emails, 1):
        result = classifier.predict(email)
        if 'error' not in result:
            preview = email[:40] + ('...' if len(email) > 40 else '')
            print(f"{i}. {result['prediction']} ({result['confidence']:.3f}) - {preview}")

def main():
    """Main pipeline execution"""
    print("🚀 SPAM CLASSIFICATION PIPELINE")
    print("=" * 40)
    
    # Load data
    df = load_cleaned_data()
    if df is None:
        raise ValueError("No cleaned data found. Run preprocessing first.")
    
    print(f"📊 Dataset: {len(df):,} samples, {df['label'].mean():.2%} spam")
    
    # Train model
    classifier = SpamClassifier()
    X, y = classifier.prepare_features(df)
    results = classifier.train(X, y)
    
    # Save and test
    classifier.save()
    test_predictions(classifier)
    
    print(f"\n🎉 Pipeline completed! AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()