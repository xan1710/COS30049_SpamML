# preprocessing.py
# Author: PixelNest Labs
# Date: 2025-10-07
# This module contains functions for preprocessing text data for spam detection.

# Import necessary libraries
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
    # Ensure necessary directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)

def detect_columns(df):
    # Auto-detect text and label columns
    text_col, label_col = None, None
    
    # Find text column by looking for common names or long text average length
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

    # Find label column by looking for common names or binary values 
    # (0/1, spam/ham, etc.)
    label_candidates = ['label', 'class', 'spam', 'target', 'y', 'category']
    for col in df.columns:
        if col.lower() in label_candidates:
            label_col = col
            break
    
    return text_col, label_col

def clean_text(s):
    # Handle NaN values 
    if pd.isna(s):
        return ''
    
    s = str(s).lower()
    s = re.sub(r'<[^>]+>', ' ', s)  # Remove HTML
    s = re.sub(r'http\S+|www\S+', ' ', s)  # Remove URLs
    s = re.sub(r'\S+@\S+', ' ', s)  # Remove emails
    s = s.replace('#', ' ').replace('@', ' ') #Remove # and @
    s = re.sub(r'[^a-z0-9\s.,!?\'-]', ' ', s) # Keep basic punctuation
    s = re.sub(r'\n', ' ', s)  # Remove newlines
    s = re.sub(r'\r', ' ', s)  # Remove carriage returns
    s = re.sub(r'\s+', ' ', s)  # Remove extra spaces
    s = re.sub(r'(\.{2,})|(-{2,})|(_{2,})|(!{4,})|(\?{3,})', ' ', s)  # Replace multiple special chars with space
    s = s.strip() # Final trim
    return s

def extract_features(text):
    # Basic checks for empty text
    if not text:
        return [0, 0, 0]
    
    # Ratio of numbers and special characters to total characters in text
    # to avoid division by zero, ensure text length > 0
    if len(text) > 0:
        number_ratio = sum(c.isdigit() for c in text) / len(text)
        special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
    else:
        number_ratio = 0
        special_ratio = 0

    # List of common spam words for counting. This can be expanded as needed,
    # since new spam words are constantly emerging.
    spam_words = ['free', 'win', 'urgent', 'money', 'click', 'now', 'offer',
                  'prize', 'deal', 'cash', 'gift', 'verify', 'guaranteed']
    # Count spam words in text (case insensitive)
    spam_count = sum(1 for word in spam_words if word in text.lower())
    
    return [number_ratio, special_ratio, spam_count]

def standardize_labels(labels):
    # Convert various label formats to binary (0: ham, 1: spam)
    if labels.dtype == 'object':
        labels = labels.astype(str).str.lower()
        spam_terms = ['1', 'spam', 'true', 'yes', 'positive']
        return labels.isin(spam_terms).astype(int)
    else:
        return (labels > 0.5).astype(int)

def preprocess_dataset(file_path):
    # Preprocess a single dataset file and return cleaned DataFrame
    df = pd.read_csv(file_path)
    
    # Auto-detect columns if not specified
    text_col, label_col = detect_columns(df)
    if not text_col or not label_col:
        return None
    
    # Create clean dataset DataFrame with cleaned text and standardized labels
    df_clean = pd.DataFrame()
    df_clean['text'] = df[text_col].apply(clean_text)
    df_clean['label'] = standardize_labels(df[label_col])
    
    # Extract features and add to DataFrame for numeric models like KNN
    features = df_clean['text'].apply(extract_features)
    df_clean[['number_ratio', 'special_char_ratio', 'spam_words']] = pd.DataFrame(features.tolist())
    
    # Add simple text stats as features for numeric models like KNN
    df_clean['text_length'] = df_clean['text'].str.len()
    df_clean['word_count'] = df_clean['text'].str.split().str.len()
    
    # Remove duplicates and empty texts
    df_clean = df_clean[df_clean['text'].str.len() > 0].drop_duplicates()
    
    return df_clean

def process_all_datasets():
    # Process all datasets in RAW_DIR and save cleaned versions
    setup_directories()
    
    # Check if RAW_DIR exists and has files to process
    if not os.path.exists(RAW_DIR):
        return
    
    # Find all CSV files in RAW_DIR to process
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not csv_files:
        return
    
    all_datasets = []
    
    # Process each dataset file, only when it exists as "cleaned" version
    for csv_file in csv_files:
        input_path = os.path.join(RAW_DIR, csv_file)
        output_path = os.path.join(CLEANED_DIR, f"cleaned_{csv_file}")
        
        df_clean = preprocess_dataset(input_path)
        
        if df_clean is not None:
            df_clean.to_csv(output_path, index=False)
            all_datasets.append(df_clean)
    
    # Create combined dataset if multiple datasets were processed
    if all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)
        
        combined_path = os.path.join(CLEANED_DIR, "combined_email_dataset.csv")
        combined_df.to_csv(combined_path, index=False)

def load_dataset(filename=None):
    # Adaptive loading of combined or individual cleaned datasets
    if filename is None:
        combined_file = os.path.join(CLEANED_DIR, "combined_email_dataset.csv")
        if os.path.exists(combined_file):
            filename = "combined_email_dataset.csv"
        else:
            # Load the first available cleaned dataset
            cleaned_files = [f for f in os.listdir(CLEANED_DIR) if f.startswith('cleaned_')]
            if not cleaned_files:
                return None
            filename = cleaned_files[0]
    
    # Load the specified cleaned dataset
    file_path = os.path.join(CLEANED_DIR, filename)
    df = pd.read_csv(file_path)
    
    return df

if __name__ == "__main__":
    print("="*40)
    print("SPAM CLASSIFIER PREPROCESSING")
    print("="*40)
    # Run preprocessing on all datasets
    process_all_datasets()
    df = load_dataset()
    if df is not None:
        print("Preprocessing completed. Sample data:")
        print(df.info())
        print(df.head())