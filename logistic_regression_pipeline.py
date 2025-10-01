# Simplified Logistic Regression Spam Classification Pipeline
# Author: Your Name
# Date: 2025-09-30

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
from preprocessing import process_dataset, normalize_text, extract_crucial_features

def load_and_combine_datasets(file_paths):
    """Load and combine multiple datasets including HuggingFace datasets"""
    dfs = []
    
    for item in file_paths:
        try:
            # Handle HuggingFace datasets
            if hasattr(item, 'to_pandas'):
                df = item.to_pandas()
                print(f"✅ Loaded HuggingFace dataset: {len(df):,} samples")
                dfs.append(df)
                
            # Handle CSV files
            elif isinstance(item, str):
                df = pd.read_csv(item, low_memory=False)
                print(f"✅ Loaded {item}: {len(df):,} samples")
                dfs.append(df)
                
        except Exception as e:
            print(f"⚠️ Could not load dataset: {e}")
    
    if not dfs:
        raise ValueError("No datasets loaded successfully")
    
    # Standardize all datasets before combining
    standardized_dfs = []
    for df in dfs:
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Find and standardize text column
        text_candidates = ['text', 'message', 'body', 'email', 'content', 'mail']
        text_col = None
        for candidate in text_candidates:
            if candidate in df.columns:
                text_col = candidate
                break
        
        if text_col and text_col != 'text':
            df['text'] = df[text_col]
        
        # Find and standardize label column
        label_candidates = ['label', 'spam', 'class', 'category', 'target']
        label_col = None
        for candidate in label_candidates:
            if candidate in df.columns:
                label_col = candidate
                break
        
        if label_col and label_col != 'label':
            df['label'] = df[label_col]
        
        # Keep essential columns
        if 'text' in df.columns and 'label' in df.columns:
            essential_cols = ['text', 'label']
            # Add existing feature columns if present
            feature_cols = ['number_ratio', 'special_char_ratio', 'sus_words_count', 'text_length', 'word_count']
            available_features = [col for col in feature_cols if col in df.columns]
            
            df_standardized = df[essential_cols + available_features]
            standardized_dfs.append(df_standardized)
    
    combined = pd.concat(standardized_dfs, ignore_index=True).drop_duplicates()
    print(f"📊 Combined dataset: {len(combined):,} samples")
    return combined

def prepare_features(df, text_col='text', max_features=1000):
    """Extract and combine TF-IDF and engineered features"""
    # TF-IDF features
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=max_features, 
        ngram_range=(1,2)
    )
    X_tfidf = vectorizer.fit_transform(df[text_col].fillna(''))
    
    # Engineered features
    feature_names = ['number_ratio', 'special_char_ratio', 'sus_words_count', 
                    'text_length', 'word_count']
    
    # Create missing features if they don't exist
    for feat in feature_names:
        if feat not in df.columns:
            if feat == 'text_length':
                df[feat] = df[text_col].str.len().fillna(0)
            elif feat == 'word_count':
                df[feat] = df[text_col].str.split().str.len().fillna(0)
            else:
                # Extract from text using preprocessing function
                features_dict = df[text_col].fillna('').apply(extract_crucial_features)
                df[feat] = features_dict.apply(lambda x: x.get(feat, 0))
    
    X_features = df[feature_names].fillna(0).values
    
    # Combine features
    X = hstack([X_tfidf, X_features])
    y = df['label'].values
    
    return X, y, vectorizer, feature_names

def train_model(X, y):
    """Train logistic regression with grid search"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Grid search
    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid, cv=3, scoring='roc_auc', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = grid.predict(X_val)
    y_test_pred = grid.predict(X_test)
    val_auc = roc_auc_score(y_val, grid.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
    
    print(f"🎯 Best params: {grid.best_params_}")
    print(f"🎯 Validation AUC: {val_auc:.4f}")
    print(f"🎯 Test AUC: {test_auc:.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Ham', 'Spam']))
    
    return grid.best_estimator_, test_auc

def predict_single_email(text, model_artifacts):
    """Predict spam probability for a single email"""
    try:
        model, vectorizer, feature_names = model_artifacts
        
        # Clean text and extract features
        clean_text = normalize_text(text)
        X_tfidf = vectorizer.transform([clean_text])
        
        # Extract numerical features
        features = extract_crucial_features(clean_text)
        features.update({
            'text_length': len(clean_text),
            'word_count': len(clean_text.split()) if clean_text else 0
        })
        
        X_features = np.array([[features.get(f, 0) for f in feature_names]])
        X_combined = hstack([X_tfidf, X_features])
        
        # Predict
        prediction = model.predict(X_combined)[0]
        probability = model.predict_proba(X_combined)[0]
        
        return {
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'spam_probability': probability[1],
            'confidence': max(probability)
        }
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}

def main():
    print("="*50)
    print("SPAM CLASSIFICATION PIPELINE")
    print("="*50)
    
    # Load datasets
    from datasets import load_dataset
    hf_dataset = load_dataset("yxzwayne/email-spam-10k", split="train")
    datasets = ['cleaned_emails.csv', 'cleaned_mail_data.csv', 'cleaned_CEAS_08.csv']
    try:
        df = load_and_combine_datasets(datasets)
    except:
        # Fallback to single dataset
        df = pd.read_csv('cleaned_emails.csv')
        print(f"📊 Using single dataset: {len(df):,} samples")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X, y, vectorizer, feature_names = prepare_features(df)
    print(f"✅ Features ready: {X.shape[1]:,} total features") # type: ignore
    
    # Train model
    print("\n🤖 Training model...")
    model, test_auc = train_model(X, y)
    
    # Save model
    model_artifacts = (model, vectorizer, feature_names)
    joblib.dump(model_artifacts, 'logistic_regression_model.pkl')
    print("💾 Model saved as 'logistic_regression_model.pkl'")
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_emails = [
        "FREE MONEY! Click here to win $10000 now! Limited time offer!",
        "Hi John, let's meet for lunch tomorrow at 12pm. Looking forward to seeing you!",
        "Dear user, you have won a prize! claim now",
        "Meeting agenda attached for project update", 
        "Urgent: verify your bank account immediately",
        "Lunch plans? Let me know your preference",
        "Free gift card available: update your info",
        "Congratulations! You've won $1 million dollars!",
        "Your account needs verification. Click here now!",
        "Team meeting scheduled for tomorrow at 3pm",
        "Invoice attached for your recent purchase",
        "Limited time offer - act now to save money!"
    ]
    
    for i, email in enumerate(test_emails, 1):
        result = predict_single_email(email, model_artifacts)
        if 'error' not in result:
            print(f"Email {i}: {result['prediction']} "
                  f"(confidence: {result['confidence']:.3f})")
    
    print(f"\n🎉 Pipeline complete! Final AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()