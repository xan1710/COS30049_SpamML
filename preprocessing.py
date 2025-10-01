# preprocessing.py
# Author: Your Name
# Date: 2025-09-30
# This module contains functions for preprocessing text data
# including handling missing values, text normalization, feature extraction,
# and adaptive dataset handling for various email datasets.


# preprocessing.py
# Simplified Email Spam Detection Preprocessing
# Maintains high performance with cleaner, more focused code

import pandas as pd
import re
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Dataset Configuration
DATA_DIR = Path("datasets")
DATASETS_DIR = DATA_DIR / "raw_datasets"
CLEANED_DIR = DATA_DIR / "cleaned_datasets"

def setup_directories():
    """Create necessary directories for datasets"""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)
    print(f"📁 Raw datasets: {DATASETS_DIR}")
    print(f"📁 Cleaned datasets: {CLEANED_DIR}")

def detect_columns(df):
    """Auto-detect text and label columns"""
    text_col, label_col = None, None
    
    # Find text column (longest average content)
    text_candidates = ['text', 'message', 'body', 'email', 'content', 'mail']
    for col in df.columns:
        if col.lower() in text_candidates or df[col].dtype == 'object':
            try:
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 50:
                    text_col = col
                    break
            except:
                continue
    
    # Find label column
    label_candidates = ['label', 'class', 'spam', 'target', 'y', 'category']
    for col in df.columns:
        if col.lower() in label_candidates:
            label_col = col
            break
    
    return text_col, label_col

def clean_text(text):
    """Clean and normalize text efficiently"""
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML
    text = re.sub(r'[^a-z0-9\s.,!?\'-]', ' ', text)  # Keep essential chars
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

def extract_spam_features(text):
    """Extract the 3 most effective spam detection features"""
    if not text:
        return [0, 0, 0]
    
    # Feature 1: Number density
    number_ratio = sum(c.isdigit() for c in text) / len(text)
    
    # Feature 2: Special character density  
    special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
    
    # Feature 3: Spam keywords count
    spam_words = ['free', 'win', 'urgent', 'money', 'click', 'now', 'offer', 
                  'prize', 'deal', 'cash', 'gift', 'verify', 'guaranteed']
    spam_count = sum(1 for word in spam_words if word in text.lower())
    
    return [number_ratio, special_ratio, spam_count]

def standardize_labels(labels):
    """Convert labels to binary (0=ham, 1=spam)"""
    if labels.dtype == 'object':
        labels = labels.astype(str).str.lower()
        spam_terms = ['1', 'spam', 'true', 'yes', 'positive']
        return labels.isin(spam_terms).astype(int)
    else:
        return (labels > 0.5).astype(int)

def preprocess_dataset(file_path):
    """Main preprocessing function - simplified but effective"""
    print(f"🔄 Processing: {os.path.basename(file_path)}")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"📊 Original shape: {df.shape}")
    
    # Auto-detect columns
    text_col, label_col = detect_columns(df)
    if not text_col or not label_col:
        print("❌ Could not detect text or label columns")
        return None
    
    print(f"📝 Text column: {text_col}")
    print(f"🏷️ Label column: {label_col}")
    
    # Create clean dataset with essential columns only
    df_clean = pd.DataFrame()
    
    # Process text
    df_clean['text'] = df[text_col].apply(clean_text)
    df_clean['label'] = standardize_labels(df[label_col])
    
    # Extract crucial features
    features = df_clean['text'].apply(extract_spam_features)
    df_clean[['number_ratio', 'special_char_ratio', 'spam_words']] = pd.DataFrame(features.tolist())
    
    # Add basic text stats
    df_clean['text_length'] = df_clean['text'].str.len()
    df_clean['word_count'] = df_clean['text'].str.split().str.len()
    
    # Remove duplicates and empty texts
    df_clean = df_clean[df_clean['text'].str.len() > 0].drop_duplicates()
    
    print(f"✅ Final shape: {df_clean.shape}")
    print(f"📊 Spam ratio: {df_clean['label'].mean():.2%}")
    
    return df_clean

def process_all_datasets():
    """Process all datasets in the datasets directory"""
    setup_directories()
    
    print("\n" + "="*60)
    print("🚀 SPAM DETECTION PREPROCESSING PIPELINE")
    print("="*60)
    
    # Check for datasets
    if not os.path.exists(DATASETS_DIR):
        print(f"📁 Please place your CSV files in: {DATASETS_DIR}")
        print("📝 Supported files: emails.csv, mail_data.csv, spam_ham.csv, etc.")
        return
    
    csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"📁 No CSV files found in: {DATASETS_DIR}")
        print("📝 Please add your email datasets to this folder")
        return
    
    processed_files = []
    
    for csv_file in csv_files:
        try:
            input_path = os.path.join(DATASETS_DIR, csv_file)
            output_file = f"cleaned_{csv_file}"
            output_path = os.path.join(CLEANED_DIR, output_file)
            
            # Process dataset
            df_clean = preprocess_dataset(input_path)
            
            if df_clean is not None:
                # Save cleaned dataset
                df_clean.to_csv(output_path, index=False)
                processed_files.append((csv_file, output_file, len(df_clean)))
                print(f"💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
    
    # Summary
    print("\n" + "-"*50)
    print("📈 PROCESSING SUMMARY")
    print("-"*50)
    
    if processed_files:
        for original, cleaned, rows in processed_files:
            print(f"✅ {original} → {cleaned} ({rows:,} rows)")
        print(f"\n🎉 Successfully processed {len(processed_files)} datasets!")
        print(f"📊 Ready for ML training with features: number_ratio, special_char_ratio, spam_words")
    else:
        print("❌ No datasets were successfully processed")

def get_feature_columns():
    """Return ML-ready feature columns"""
    return ['number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']

def load_cleaned_data(filename=None):
    """Load processed dataset for ML training"""
    if filename is None:
        # Get the first available cleaned file
        cleaned_files = [f for f in os.listdir(CLEANED_DIR) if f.startswith('cleaned_')]
        if not cleaned_files:
            print("❌ No cleaned datasets found. Run process_all_datasets() first.")
            return None
        filename = cleaned_files[0]
    
    file_path = os.path.join(CLEANED_DIR, filename)
    df = pd.read_csv(file_path)
    
    print(f"📊 Loaded: {filename}")
    print(f"📊 Shape: {df.shape}")
    print(f"📊 Features: {get_feature_columns()}")
    
    return df

# Example usage and testing
if __name__ == "__main__":
    # Print directory structure
    print("📁 DATASET ORGANIZATION:")
    print(f"   Raw datasets → {DATASETS_DIR}")
    print(f"   Cleaned data → {CLEANED_DIR}")
    print("\n💡 USAGE:")
    print("   1. Place CSV files in datasets/ folder")
    print("   2. Run: process_all_datasets()")
    print("   3. Use: load_cleaned_data() for ML training")
    
    # Process datasets if any exist
    process_all_datasets()
    
    # Demo with sample data if no real datasets
    try:
        df = load_cleaned_data()
        if df is not None:
            print("\n📋 Sample cleaned data:")
            print(df.head())
            print(f"\n🎯 ML Features ready: {get_feature_columns()}")
    except:
        print("\n💡 Add your email datasets to get started!")