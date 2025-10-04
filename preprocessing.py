# preprocessing.py
# Author: Your Name
# Date: 2025-09-30
# This module contains functions for preprocessing text data
# including handling missing values, text normalization, feature extraction,
# and adaptive dataset handling for various email datasets.

import pandas as pd
import re
import numpy as np
import os
from pathlib import Path

# Dataset Configuration
DATA_DIR = Path("datasets")
RAW_DIR = DATA_DIR / "raw_datasets"
CLEANED_DIR = DATA_DIR / "cleaned_datasets"

def setup_directories():
    """Create necessary directories for datasets"""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)

def detect_columns(df):
    """Auto-detect text and label columns"""
    text_col, label_col = None, None
    
    # Find text column
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
    # Handle NaN values
    if pd.isna(text):
        return ''
    
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = text.replace('#', ' ').replace('@', ' ') #Remove # and @
    text = re.sub(r'[^a-z0-9\s.,!?\'-]', ' ', text) # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    text = re.sub(r'\.{2,}', ' ', text)  # Replace multiple dots with space
    text = re.sub(r'-{2,}', ' ', text)  # Replace multiple hyphens with space
    text = re.sub(r'_{2,}', ' ', text)  # Replace multiple underscores with space
    text = re.sub(r'!{4,}', '!', text)  # Replace multiple exclamation marks
    text = re.sub(r'\?{3,}', '?', text)  # Replace multiple question marks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

def extract_features(text):
    """Extract spam detection features"""
    # Basic checks
    if not text:
        return [0, 0, 0]
    
    # Ratio of numbers and special characters
    number_ratio = sum(c.isdigit() for c in text) / len(text)
    special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)

    # List of spam words
    spam_words = ['free', 'win', 'urgent', 'money', 'click', 'now', 'offer',
                  'prize', 'deal', 'cash', 'gift', 'verify', 'guaranteed']
    # Count spam words
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
    """Main preprocessing function"""
    df = pd.read_csv(file_path)
    
    # Auto-detect columns
    text_col, label_col = detect_columns(df)
    if not text_col or not label_col:
        return None
    
    # Create clean dataset
    df_clean = pd.DataFrame()
    df_clean['text'] = clean_text(df[text_col])
    df_clean['label'] = standardize_labels(df[label_col])
    
    # Extract features
    features = df_clean['text'].apply(extract_features)
    df_clean[['number_ratio', 'special_char_ratio', 'spam_words']] = pd.DataFrame(features.tolist())
    
    # Add text stats
    df_clean['text_length'] = df_clean['text'].str.len()
    df_clean['word_count'] = df_clean['text'].str.split().str.len()
    
    # Remove duplicates and empty texts
    df_clean = df_clean[df_clean['text'].str.len() > 0].drop_duplicates()
    
    return df_clean

def process_all_datasets():
    """Process all datasets in the datasets directory"""
    setup_directories()
    
    if not os.path.exists(RAW_DIR):
        return
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not csv_files:
        return
    
    all_datasets = []
    
    # Process each dataset
    for csv_file in csv_files:
        try:
            input_path = os.path.join(RAW_DIR, csv_file)
            output_path = os.path.join(CLEANED_DIR, f"cleaned_{csv_file}")
            
            df_clean = preprocess_dataset(input_path)
            
            if df_clean is not None:
                df_clean.to_csv(output_path, index=False)
                all_datasets.append(df_clean)
                
        except Exception as e:
            continue
    
    # Create combined dataset
    if all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)
        
        combined_path = os.path.join(CLEANED_DIR, "combined_email_dataset.csv")
        combined_df.to_csv(combined_path, index=False)

def load_cleaned_data(filename=None):
    """Load processed dataset for ML training"""
    # Adaptive loading of combined or individual cleaned datasets
    if filename is None:
        combined_file = os.path.join(CLEANED_DIR, "combined_email_dataset.csv")
        if os.path.exists(combined_file):
            filename = "combined_email_dataset.csv"
        else:
            cleaned_files = [f for f in os.listdir(CLEANED_DIR) if f.startswith('cleaned_')]
            if not cleaned_files:
                return None
            filename = cleaned_files[0]
    
    file_path = os.path.join(CLEANED_DIR, filename)
    df = pd.read_csv(file_path)
    
    return df

if __name__ == "__main__":
    print("="*40)
    print("🚀 SPAM CLASSIFIER PREPROCESSING")
    print("="*40)
    process_all_datasets()
    df = load_cleaned_data()
    if df is not None:
        print("Preprocessing completed. Sample data:")
        print(df.info())
        print(df.head())