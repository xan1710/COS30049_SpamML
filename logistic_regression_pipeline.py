import pandas as pd
import numpy as np
from datasets import load_dataset
from preprocessing import process_dataset, preprocess_email_data
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

# # --- Data Preprocessing & Feature Engineering ---

# # def clean_text(text):
# #     # Lowercase
# #     text = text.lower()
# #     # Remove special characters except ',', '.', '?', '!'
# #     text = re.sub(r"[^a-z0-9,.!? ]+", " ", text)
# #     # Remove extra spaces
# #     text = re.sub(r"\s+", " ", text).strip()
# #     return text

# # def number_ratio(text):
# #     # Ratio of digits to total length
# #     num_digits = sum(c.isdigit() for c in text)
# #     return num_digits / max(len(text), 1)

# # def char_ratio(text):
# #     # Ratio of alphabetic chars to total length
# #     num_chars = sum(c.isalpha() for c in text)
# #     return num_chars / max(len(text), 1)

# # # Load dataset
# # # df = pd.read_csv("hf://datasets/bourigue/data_email_spam/spam_email_dataset.csv", low_memory=False)
# # df = pd.read_csv('emails.csv', low_memory=False)

# # # Remove rows with missing values in key columns
# # df = df.dropna(subset=['label', 'text'])
# # df['text'] = df['text'].astype(str)

# # # Make sure labels are binary (0 and 1)
# # df['label'] = pd.to_numeric(df['label'], errors='coerce')
# # df = df[df['label'].isin([0, 1])]

# # # Clean text
# # df['clean_text'] = df['text'].apply(clean_text)

# # # Feature engineering
# # df['number_ratio'] = df['clean_text'].apply(number_ratio)
# # df['char_ratio'] = df['clean_text'].apply(char_ratio)

# # df = process_dataset('emails.csv')

# # # Save cleaned dataset
# # df.to_csv('cleaned_emails.csv', index=False)

# # df = process_dataset('emails.csv')
# df = process_dataset("hf://datasets/bourigue/data_email_spam/spam_email_dataset.csv")

# # --- TF-IDF Vectorization ---
# tfidf = TfidfVectorizer(
#     stop_words='english',
#     max_features=1000,
#     ngram_range=(1,2),
#     lowercase=True
# )
# X_tfidf = tfidf.fit_transform(df['clean_text'])

# # # Combine TF-IDF features with engineered features
# # import numpy as np
# # X_extra = df[['number_ratio', 'char_ratio']].values
# # from scipy.sparse import hstack
# # X = hstack([X_tfidf, X_extra])
# # y = df['label']

# # Combine TF-IDF features with engineered features from preprocessing
# feature_columns = [
#     'text_length', 
#     'word_count', 
#     'avg_word_length',
#     'number_ratio',
#     'special_char_ratio',
#     'uppercase_ratio'
# ]
# X_extra = df[feature_columns].values
# X = hstack([X_tfidf, X_extra])
# y = df['label']

# # --- Train/Validation/Test Split ---
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42, shuffle=True)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

# # --- Model Training & Hyperparameter Tuning ---
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'penalty': ['l2'],
#     'solver': ['lbfgs', 'liblinear']
# }
# logreg = LogisticRegression(max_iter=5000, random_state=42)
# grid = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc')
# grid.fit(X_train, y_train)

# # print("\nBest parameters:", grid.best_params_)
# # print("Best ROC AUC on validation:", grid.best_score_)

# # # --- Validation Evaluation ---
# # y_val_pred = grid.predict(X_val)
# # y_val_proba = grid.predict_proba(X_val)[:, 1]
# # print("\nValidation Classification Report:")
# # print(classification_report(y_val, y_val_pred))
# # print("Validation Confusion Matrix:")
# # print(confusion_matrix(y_val, y_val_pred))
# # print("Validation ROC AUC:", roc_auc_score(y_val, y_val_proba))

# # # --- Test Evaluation ---
# # y_test_pred = grid.predict(X_test)
# # y_test_proba = grid.predict_proba(X_test)[:, 1]
# # print("\nTest Classification Report:")
# # print(classification_report(y_test, y_test_pred))
# # print("Test Confusion Matrix:")
# # print(confusion_matrix(y_test, y_test_pred))
# # print("Test ROC AUC:", roc_auc_score(y_test, y_test_proba))

# # # --- Save Model & Vectorizer ---
# # model_artifacts = {
# #     'model': grid.best_estimator_,
# #     'vectorizer': tfidf,
# #     'feature_columns': feature_columns
# # }
# # joblib.dump(model_artifacts, 'logistic_regression_model.pkl')
# # joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# # --- Model Training & Hyperparameter Tuning ---
# print("\n" + "="*50)
# print("MODEL TRAINING RESULTS")
# print("="*50)
# print(f"Best parameters: {grid.best_params_}")
# print(f"Best ROC AUC on validation: {grid.best_score_:.4f}")

# # --- Validation Evaluation ---
# print("\n" + "="*50)
# print("VALIDATION SET EVALUATION")
# print("="*50)
# y_val_pred = grid.predict(X_val)
# y_val_proba = grid.predict_proba(X_val)[:, 1]
# print("\nClassification Report:")
# print("-"*30)
# print(classification_report(y_val, y_val_pred))
# print("\nConfusion Matrix:")
# print("-"*30)
# print(confusion_matrix(y_val, y_val_pred))
# print(f"\nROC AUC Score: {roc_auc_score(y_val, y_val_proba):.4f}")

# # --- Test Evaluation ---
# print("\n" + "="*50)
# print("TEST SET EVALUATION")
# print("="*50)
# y_test_pred = grid.predict(X_test)
# y_test_proba = grid.predict_proba(X_test)[:, 1]
# print("\nClassification Report:")
# print("-"*30)
# print(classification_report(y_test, y_test_pred))
# print("\nConfusion Matrix:")
# print("-"*30)
# print(confusion_matrix(y_test, y_test_pred))
# print(f"\nROC AUC Score: {roc_auc_score(y_test, y_test_proba):.4f}")

# # --- Save Model & Vectorizer ---
# print("\n" + "="*50)
# print("SAVING MODEL ARTIFACTS")
# print("="*50)
# model_artifacts = {
#     'model': grid.best_estimator_,
#     'vectorizer': tfidf,
#     'feature_columns': feature_columns
# }
# joblib.dump(model_artifacts, 'logistic_regression_model.pkl')
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
# print("Model and vectorizer saved successfully!")
# print("="*50 + "\n")





# --- Data Loading & Preprocessing ---
# print("="*60)
# print("EMAIL SPAM CLASSIFICATION PIPELINE")
# print("="*60)

# # Process dataset (will auto-detect columns and create synthetic data if file not found)
# input_file = 'emails.csv'
# cleaned_df = process_dataset(input_file)

# print(f"\nDataset processed successfully!")
# print(f"Total samples: {len(cleaned_df)}")
# print(f"Spam ratio: {cleaned_df['label'].mean():.2%}")

# # --- Feature Engineering & Vectorization ---
# print("\n" + "="*50)
# print("FEATURE ENGINEERING")
# print("="*50)

# # TF-IDF on cleaned text
# tfidf = TfidfVectorizer(
#     stop_words='english',
#     max_features=1000,
#     ngram_range=(1,2),
#     lowercase=True
# )
# X_tfidf = tfidf.fit_transform(cleaned_df['clean_text'])

# # Combine with engineered features
# feature_columns = [
#     'text_length', 'number_ratio',
#     'special_char_ratio', 'uppercase_ratio', 'susp_word_count',
#     'has_suspicious', 'num_links', 'num_emails', 'num_phones',
#     'num_exclamations', 'num_questions', 'caps_ratio'
# ]

# X_extra = cleaned_df[feature_columns].values
# X = hstack([X_tfidf, X_extra])
# y = cleaned_df['label']

# print(f"TF-IDF features: {X_tfidf.shape[1]}")
# print(f"Additional features: {len(feature_columns)}")
# print(f"Total features: {X.shape[1]}")

# # --- Train/Validation/Test Split ---
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True
# )
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp, shuffle=True
# )

# print(f"\nTrain set: {X_train.shape[0]} samples")
# print(f"Validation set: {X_val.shape[0]} samples") 
# print(f"Test set: {X_test.shape[0]} samples")

# # --- Model Training & Hyperparameter Tuning ---
# print("\n" + "="*50)
# print("MODEL TRAINING")
# print("="*50)

# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'penalty': ['l2'],
#     'solver': ['lbfgs', 'liblinear']
# }

# logreg = LogisticRegression(max_iter=1000, random_state=42)
# grid = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid.fit(X_train, y_train)

# print(f"Best parameters: {grid.best_params_}")
# print(f"Best cross-validation ROC AUC: {grid.best_score_:.4f}")

# # --- Evaluation ---
# print("\n" + "="*50)
# print("VALIDATION SET EVALUATION")
# print("="*50)

# y_val_pred = grid.predict(X_val)
# y_val_proba = grid.predict_proba(X_val)[:, 1]

# print("Classification Report:")
# print("-" * 30)
# print(classification_report(y_val, y_val_pred))
# print(f"\nROC AUC Score: {roc_auc_score(y_val, y_val_proba):.4f}")

# print("\n" + "="*50)
# print("TEST SET EVALUATION")
# print("="*50)

# y_test_pred = grid.predict(X_test)
# y_test_proba = grid.predict_proba(X_test)[:, 1]

# print("Classification Report:")
# print("-" * 30)
# print(classification_report(y_test, y_test_pred))
# print(f"\nROC AUC Score: {roc_auc_score(y_test, y_test_proba):.4f}")

# # --- Save Model ---
# print("\n" + "="*50)
# print("SAVING MODEL")
# print("="*50)

# model_artifacts = {
#     'model': grid.best_estimator_,
#     'vectorizer': tfidf,
#     'feature_columns': feature_columns
# }

# joblib.dump(model_artifacts, 'spam_classifier_complete.pkl')
# print("Model saved successfully as 'spam_classifier_complete.pkl'")
# print("="*60)






# import pandas as pd
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')

# # Core ML libraries
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import StandardScaler
# from scipy.sparse import hstack
# import joblib

# # Custom preprocessing
# from preprocessing import process_dataset

# # Visualization
# import matplotlib.pyplot as plt
# import seaborn as sns

# def create_logistic_regression_pipeline():
#     """Complete logistic regression pipeline for email spam classification."""
    
#     print("="*60)
#     print("LOGISTIC REGRESSION SPAM CLASSIFICATION PIPELINE")
#     print("="*60)
    
#     # --- 1. Data Loading & Preprocessing ---
#     print("\n🔄 STEP 1: DATA LOADING & PREPROCESSING")
#     print("-" * 50)
    
#     # Try multiple data sources
#     data_sources = [
#         'emails.csv',
#         'combined_data.csv',
#         'hf://datasets/bourigue/data_email_spam/spam_email_dataset.csv',
#     ]
    
#     cleaned_df = None
#     for source in data_sources:
#         try:
#             cleaned_df = process_dataset(source)
#             print(f"✅ Successfully loaded data from: {source}")
#             break
#         except Exception as e:
#             print(f"❌ Failed to load {source}: {str(e)}")
#             continue
    
#     if cleaned_df is None:
#         print("🔧 Creating synthetic dataset for demonstration...")
#         cleaned_df = process_dataset('synthetic_data')
    
#     print(f"📊 Dataset Summary:")
#     print(f"   • Total samples: {len(cleaned_df):,}")
#     print(f"   • Spam samples: {cleaned_df['label'].sum():,} ({cleaned_df['label'].mean():.1%})")
#     print(f"   • Ham samples: {(cleaned_df['label'] == 0).sum():,} ({(cleaned_df['label'] == 0).mean():.1%})")
    
#     # --- 2. Feature Engineering ---
#     print("\n🔧 STEP 2: FEATURE ENGINEERING")
#     print("-" * 50)
    
#     # TF-IDF Vectorization
#     print("Creating TF-IDF features...")
#     tfidf = TfidfVectorizer(
#         stop_words='english',
#         max_features=5000,
#         ngram_range=(1, 2),
#         lowercase=True,
#         min_df=2,
#         max_df=0.95
#     )
    
#     X_tfidf = tfidf.fit_transform(cleaned_df['clean_text'])
    
#     # Numerical features
#     feature_columns = [
#         'text_length', 'number_ratio', 'special_char_ratio', 
#         'uppercase_ratio', 'num_links', 'num_emails', 'num_phones', 
#         'num_exclamations', 'num_questions', 'caps_ratio'
#     ]
    
#     # Ensure all feature columns exist
#     missing_cols = [col for col in feature_columns if col not in cleaned_df.columns]
#     if missing_cols:
#         print(f"⚠️  Missing columns: {missing_cols}")
#         # Fill missing columns with zeros
#         for col in missing_cols:
#             cleaned_df[col] = 0
    
#     X_numerical = cleaned_df[feature_columns].values
    
#     # Scale numerical features
#     scaler = StandardScaler()
#     X_numerical_scaled = scaler.fit_transform(X_numerical)
    
#     # Combine features
#     X = hstack([X_tfidf, X_numerical_scaled])
#     y = cleaned_df['label'].values
    
#     print(f"✅ Feature Engineering Complete:")
#     print(f"   • TF-IDF features: {X_tfidf.shape[1]:,}")
#     print(f"   • Numerical features: {len(feature_columns)}")
#     # print(f"   • Total features: {X.shape[1]:,}")
#     print(f"   • Feature matrix shape: {X.shape}")
    
#     # --- 3. Train/Validation/Test Split ---
#     print("\n📊 STEP 3: DATA SPLITTING")
#     print("-" * 50)
    
#     # First split: separate test set (20%)
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         X, y, test_size=0.1, random_state=0, shuffle=True
#     )
    
#     # Second split: separate train and validation (80% of remaining)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp, shuffle=True
#     )
    
#     print(f"📈 Data Split Summary:")
#     print(f"   • Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(y):.1%})")
#     print(f"   • Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(y):.1%})")
#     print(f"   • Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(y):.1%})")
    
#     # --- 4. Model Training & Hyperparameter Tuning ---
#     print("\n🤖 STEP 4: MODEL TRAINING & HYPERPARAMETER TUNING")
#     print("-" * 50)
    
#     # Define hyperparameter grid
#     param_grid = {
#         'C': [0.01, 0.1, 1, 10, 100],
#         'penalty': ['l1', 'l2'],
#         'solver': ['liblinear'],  # Works with both L1 and L2
#         'max_iter': [5000]
#     }
    
#     # Create base model
#     logreg = LogisticRegression(random_state=42, class_weight='balanced')
    
#     # Grid search with cross-validation
#     print("🔍 Performing grid search with 5-fold cross-validation...")
#     grid_search = GridSearchCV(
#         estimator=logreg,
#         param_grid=param_grid,
#         cv=5,
#         scoring='roc_auc',
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X_train, y_train)
    
#     print(f"✅ Best Parameters: {grid_search.best_params_}")
#     print(f"✅ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
#     # Get best model
#     best_model = grid_search.best_estimator_
    
#     # --- 5. Model Evaluation ---
#     print("\n📈 STEP 5: MODEL EVALUATION")
#     print("-" * 50)
    
#     # Validation set evaluation
#     print("🔍 VALIDATION SET PERFORMANCE:")
#     y_val_pred = best_model.predict(X_val)
#     y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
#     print("\nClassification Report:")
#     print(classification_report(y_val, y_val_pred, target_names=['Ham', 'Spam']))
    
#     val_auc = roc_auc_score(y_val, y_val_proba)
#     print(f"ROC-AUC Score: {val_auc:.4f}")
    
#     # Test set evaluation
#     print("\n🎯 TEST SET PERFORMANCE:")
#     y_test_pred = best_model.predict(X_test)
#     y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_test_pred, target_names=['Ham', 'Spam']))
    
#     test_auc = roc_auc_score(y_test, y_test_proba)
#     print(f"ROC-AUC Score: {test_auc:.4f}")
    
#     # Confusion Matrix
#     print("\nConfusion Matrix (Test Set):")
#     cm = confusion_matrix(y_test, y_test_pred)
#     print(cm)
    
#     # --- 6. Feature Importance Analysis ---
#     print("\n🔍 STEP 6: FEATURE IMPORTANCE ANALYSIS")
#     print("-" * 50)
    
#     # Get feature names
#     tfidf_features = [f"tfidf_{i}" for i in range(X_tfidf.shape[1])]
#     all_feature_names = tfidf_features + feature_columns
    
#     # Get coefficients
#     coefficients = best_model.coef_[0]
    
#     # Create feature importance dataframe
#     feature_importance = pd.DataFrame({
#         'feature': all_feature_names,
#         'coefficient': coefficients,
#         'abs_coefficient': np.abs(coefficients)
#     }).sort_values('abs_coefficient', ascending=False)
    
#     print("Top 10 Most Important Features:")
#     print(feature_importance.head(10)[['feature', 'coefficient']].to_string(index=False))
    
#     # --- 7. Save Model Artifacts ---
#     print("\n💾 STEP 7: SAVING MODEL ARTIFACTS")
#     print("-" * 50)
    
#     model_artifacts = {
#         'model': best_model,
#         'vectorizer': tfidf,
#         'scaler': scaler,
#         'feature_columns': feature_columns,
#         'best_params': grid_search.best_params_,
#         'cv_score': grid_search.best_score_,
#         'test_auc': test_auc,
#         'feature_importance': feature_importance
#     }
    
#     # Save complete model
#     joblib.dump(model_artifacts, 'spam_classifier_complete.pkl')
#     print("✅ Model saved as 'spam_classifier_complete.pkl'")
    
#     # Save individual components for easier loading
#     joblib.dump(best_model, 'logistic_regression_model.pkl')
#     joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
#     joblib.dump(scaler, 'feature_scaler.pkl')
    
#     print("✅ Individual components saved:")
#     print("   • logistic_regression_model.pkl")
#     print("   • tfidf_vectorizer.pkl") 
#     print("   • feature_scaler.pkl")
    
#     print(f"\n🎉 PIPELINE COMPLETE!")
#     print(f"Final Test ROC-AUC: {test_auc:.4f}")
#     print("="*60)
    
#     return model_artifacts

# # --- 8. Prediction Function ---
# def predict_email(text: str, model_path: str = 'spam_classifier_complete.pkl') -> dict:
#     """Predict if an email is spam or ham."""
#     try:
#         # Load model artifacts
#         artifacts = joblib.load(model_path)
#         model = artifacts['model']
#         vectorizer = artifacts['vectorizer']
#         scaler = artifacts['scaler']
#         feature_columns = artifacts['feature_columns']
        
#         # Import preprocessing functions
#         from preprocessing import clean_text, extract_spam_features
        
#         # Preprocess text
#         clean_email = clean_text(text)
        
#         # Extract TF-IDF features
#         X_tfidf = vectorizer.transform([clean_email])
        
#         # Extract numerical features
#         features = extract_spam_features(text)
#         X_numerical = np.array([[features.get(col, 0) for col in feature_columns]])
#         X_numerical_scaled = scaler.transform(X_numerical)
        
#         # Combine features
#         X_combined = hstack([X_tfidf, X_numerical_scaled])
        
#         # Make prediction
#         prediction = model.predict(X_combined)[0]
#         probability = model.predict_proba(X_combined)[0]
        
#         return {
#             'prediction': 'Spam' if prediction == 1 else 'Ham',
#             'spam_probability': probability[1],
#             'ham_probability': probability[0],
#             'confidence': max(probability)
#         }
        
#     except Exception as e:
#         return {'error': f"Prediction failed: {str(e)}"}

# if __name__ == "__main__":
#     # Run the complete pipeline
#     model_artifacts = create_logistic_regression_pipeline()
    
#     # Test the prediction function
#     print("\n🧪 TESTING PREDICTION FUNCTION")
#     print("-" * 50)
    
#     test_emails = [
#         "FREE MONEY! Click here to win $10000 now! Limited time offer!",
#         "Hi John, let's meet for lunch tomorrow at 12pm. Looking forward to seeing you!",
#         "Dear user, you have won a prize! claim now",
#         "Meeting agenda attached for project update", 
#         "Urgent: verify your bank account immediately",
#         "Lunch plans? Let me know your preference",
#         "Free gift card available: update your info",
#         "Congratulations! You've won $1 million dollars!",
#         "Your account needs verification. Click here now!",
#         "Team meeting scheduled for tomorrow at 3pm",
#         "Invoice attached for your recent purchase",
#         "Limited time offer - act now to save money!"
#     ]
    
#     for i, email in enumerate(test_emails, 1):
#         result = predict_email(email)
#         print(f"\nTest Email {i}: {email[:50]}...")
#         print(f"Prediction: {result.get('prediction', 'Error')}")
#         print(f"Confidence: {result.get('confidence', 0):.3f}")




import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
from preprocessing import process_dataset, clean_text, extract_spam_features, basic_text_features


def predict_single_email(text, model_artifacts):
    """Predict if a single email is spam or ham"""
    try:
        # Extract components
        model = model_artifacts['model']
        vectorizer = model_artifacts['vectorizer']
        feature_columns = model_artifacts['feature_columns']
        
        # Preprocess text
        clean_email = clean_text(text)
        
        # Extract TF-IDF features
        X_tfidf = vectorizer.transform([clean_email])
        
        # Extract numerical features
        spam_features = extract_spam_features(text)
        basic_features = basic_text_features(clean_email)
        
        # Combine all features
        all_features = {**spam_features, **basic_features}
        X_numerical = np.array([[all_features.get(col, 0) for col in feature_columns]])
        
        # Combine TF-IDF and numerical features
        X_combined = hstack([X_tfidf, X_numerical])
        
        # Make prediction
        prediction = model.predict(X_combined)[0]
        probability = model.predict_proba(X_combined)[0]
        
        return {
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'spam_probability': probability[1],
            'ham_probability': probability[0],
            'confidence': max(probability)
        }
        
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}

# --- Main Pipeline ---
print("="*60)
print("LOGISTIC REGRESSION SPAM CLASSIFICATION PIPELINE")
print("="*60)

# --- Data Loading & Preprocessing ---
print("\n🔄 STEP 1: DATA LOADING & PREPROCESSING")
print("-" * 50)

try:
    cleaned_df = process_dataset('emails.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Total samples: {len(cleaned_df):,}")
    print(f"📊 Spam ratio: {cleaned_df['label'].mean():.2%}")
    print(f"📊 Spam samples: {cleaned_df['label'].sum():,} ({cleaned_df['label'].mean():.1%})")
    print(f"📊 Ham samples: {(cleaned_df['label'] == 0).sum():,}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# --- Feature Engineering & Vectorization ---
print("\n🔧 STEP 2: FEATURE ENGINEERING")
print("-" * 50)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1,2),
    lowercase=True
)
X_tfidf = tfidf.fit_transform(cleaned_df['clean_text'])

# Combine with engineered features
feature_columns = [
    'text_length', 'word_count', 'avg_word_length', 'number_ratio',
    'special_char_ratio', 'uppercase_ratio', 'susp_word_count',
    'has_suspicious', 'num_links', 'num_emails', 'num_phones',
    'num_exclamations', 'num_questions', 'caps_ratio'
]

# Ensure all feature columns exist
missing_cols = [col for col in feature_columns if col not in cleaned_df.columns]
if missing_cols:
    print(f"⚠️ Missing columns: {missing_cols}")
    for col in missing_cols:
        cleaned_df[col] = 0

X_extra = cleaned_df[feature_columns].values
X = hstack([X_tfidf, X_extra])
y = cleaned_df['label']

print(f"✅ TF-IDF features: {X_tfidf.shape[1]:,}")
print(f"✅ Additional features: {len(feature_columns)}")
# print(f"✅ Total features: {X.shape[1]:,}")

# --- Train/Validation/Test Split ---
print("\n📊 STEP 3: DATA SPLITTING")
print("-" * 50)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp, shuffle=True
)

print(f"📈 Training set: {X_train.shape[0]:,} samples")
print(f"📈 Validation set: {X_val.shape[0]:,} samples") 
print(f"📈 Test set: {X_test.shape[0]:,} samples")

# --- Model Training & Hyperparameter Tuning ---
print("\n🤖 STEP 4: MODEL TRAINING")
print("-" * 50)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

logreg = LogisticRegression(max_iter=50000, random_state=42)
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"🎯 Best parameters: {grid.best_params_}")
print(f"🎯 Best cross-validation ROC AUC: {grid.best_score_:.4f}")

# --- Model Evaluation ---
print("\n📈 STEP 5: MODEL EVALUATION")
print("-" * 50)

# Validation evaluation
y_val_pred = grid.predict(X_val)
y_val_proba = grid.predict_proba(X_val)[:, 1]

print("🔍 VALIDATION SET PERFORMANCE:")
print("Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Ham', 'Spam']))
print(f"ROC AUC Score: {roc_auc_score(y_val, y_val_proba):.4f}")

# Test evaluation
y_test_pred = grid.predict(X_test)
y_test_proba = grid.predict_proba(X_test)[:, 1]

print("\n🎯 TEST SET PERFORMANCE:")
print("Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Ham', 'Spam']))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_test_proba):.4f}")

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# --- Save Model ---
print("\n💾 STEP 6: SAVING MODEL")
print("-" * 50)

model_artifacts = {
    'model': grid.best_estimator_,  
    'vectorizer': tfidf,
    'feature_columns': feature_columns,
    'best_params': grid.best_params_,
    'test_auc': roc_auc_score(y_test, y_test_proba)
}

joblib.dump(model_artifacts, 'spam_classifier_complete.pkl')
print("✅ Model saved as 'spam_classifier_complete.pkl'")

# --- Test with Sample Emails ---
print("\n🧪 STEP 7: TESTING WITH SAMPLE EMAILS")
print("=" * 50)

sample_emails = [
        "FREE MONEY! You have won $1,000,000! Click here now to claim your prize!",
        "URGENT! Your bank account needs immediate verification! Click this link now!",
        "Hi John, thanks for the meeting today. The project timeline looks good.",
        "Meeting reminder: Team standup tomorrow at 10 AM in conference room B.",
        "Congratulations! You are our lucky winner! Claim your FREE gift card!",
        "Invoice #12345 for your recent order is attached. Payment due within 30 days.",
        "Mom, can you pick me up from school at 3 PM today? Soccer practice was cancelled.",
        "MAKE MONEY FAST! Work from home and earn $5000/week! No experience needed!"
    ]

print(f"Testing model with {len(sample_emails)} sample emails:\n")

correct_predictions = 0
for i, email_data in enumerate(sample_emails, 1):
    result = predict_single_email(email_data, model_artifacts)
    
    if 'error' in result:
        print(f"❌ Email {i}: Error - {result['error']}")
        continue
        
    prediction = result['prediction']
    confidence = result['confidence']
    is_correct = prediction
    
    if is_correct:
        correct_predictions += 1
        status = "✅ Correct"
    else:
        status = "❌ Incorrect"

    print(f"Email {i}: {email_data[:50]}...")
    print(f"   Prediction: {prediction} | {status}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Spam Prob: {result['spam_probability']:.3f} | Ham Prob: {result['ham_probability']:.3f}")
    print()

# Summary
accuracy = correct_predictions / len(sample_emails)
print(f"📊 Sample Email Test Results:")
print(f"   Correct Predictions: {correct_predictions}/{len(sample_emails)}")
print(f"   Accuracy: {accuracy:.1%}")

print(f"\n🎉 PIPELINE COMPLETE!")
print(f"Final Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
print("=" * 60)