# preprocessing.py
# Author: Your Name
# Date: 2025-09-30
# This module contains functions for preprocessing text data
# including handling missing values, text normalization, feature extraction,
# and adaptive dataset handling for various email datasets.


import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import os
import warnings
warnings.filterwarnings('ignore')

def detect_dataset_structure(df):
    """
    Automatically detect the structure and type of the dataset
    Returns dataset info for adaptive preprocessing
    """
    info = {
        'dataset_type': 'unknown',
        'text_col': None,
        'label_col': None,
        'columns': list(df.columns),
        'shape': df.shape,
        'missing_data': df.isnull().sum().to_dict()
    }
    
    # Detect text column (long text content)
    text_candidates = ['text', 'message', 'body', 'email', 'content', 'mail', 'subject']
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        # Check for exact matches first
        if col_lower in text_candidates:
            info['text_col'] = col
            break
        
        # Check for columns with long average text length
        if df[col].dtype == 'object':
            try:
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 50:  # Likely contains substantial text
                    info['text_col'] = col
                    break
            except:
                continue
    
    # Detect label column
    label_candidates = ['label', 'class', 'spam', 'category', 'target', 'y']
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in label_candidates:
            info['label_col'] = col
            break
    
    # Determine dataset type
    if info['text_col'] and info['label_col']:
        info['dataset_type'] = 'text_classification'
    elif info['text_col']:
        info['dataset_type'] = 'text_only'
    else:
        info['dataset_type'] = 'structured'
    
    return info

def handle_missing_values(df, strategy='adaptive'):
    """
    Handle missing values based on column type and data distribution
    """
    print("🔧 Handling missing values...")
    
    missing_summary = df.isnull().sum()
    if missing_summary.sum() == 0:
        print("✅ No missing values found")
        return df
    
    print(f"📊 Missing values summary:")
    for col, missing_count in missing_summary.items():
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            print(f"   • {col}: {missing_count} ({missing_pct:.1f}%)")
    
    df_cleaned = df.copy()
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        
        if missing_count == 0:
            continue
            
        missing_pct = (missing_count / len(df)) * 100
        
        # Drop columns with >50% missing data
        if missing_pct > 50:
            print(f"⚠️ Dropping column '{col}' (>{missing_pct:.1f}% missing)")
            df_cleaned = df_cleaned.drop(columns=[col])
            continue
        
        # Handle based on data type
        if df[col].dtype in ['object', 'string']:
            # Text/categorical data - fill with 'unknown' or most frequent
            if missing_pct < 10:
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'unknown'
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
            else:
                df_cleaned[col] = df_cleaned[col].fillna('unknown')
        
        elif df[col].dtype in ['int64', 'float64']:
            # Numeric data - fill with median or mean
            if df[col].skew() > 1:  # Right-skewed data
                df_cleaned[col] = df_cleaned[col].fillna(df[col].median())
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df[col].mean())
    
    return df_cleaned

def normalize_text(text):
    """
    Comprehensive text cleaning and normalization
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-z0-9\s.,!?\'"-]', ' ', text)
    
    # Normalize repeated punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{2,}', '.', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def standardize_labels(labels):
    """
    Convert various label formats to standardized binary format
    """
    if labels.dtype == 'object':
        labels = labels.astype(str).str.lower().str.strip()
        
        # Map spam indicators
        spam_indicators = ['1', 'spam', 'true', 'yes', '1.0', 'positive']
        ham_indicators = ['0', 'ham', 'false', 'no', '0.0', 'negative']
        
        def convert_label(x):
            if x in spam_indicators:
                return 1
            elif x in ham_indicators:
                return 0
            else:
                try:
                    val = float(x)
                    return 1 if val > 0.5 else 0
                except:
                    return 0
        
        return labels.apply(convert_label)
    else:
        # Already numeric, just ensure binary
        return (labels > 0.5).astype(int)

def extract_crucial_features(cleaned_text):
    """
    Extract the 3 most crucial features for spam detection from cleaned text
    """
    
    # 1. NUMBER RATIO
    if not cleaned_text:
        number_ratio = 0
    else:
        number_count = sum(c.isdigit() for c in cleaned_text)
        number_ratio = number_count / len(cleaned_text)
    
    # 2. SPECIAL CHARACTER RATIO
    if not cleaned_text:
        special_char_ratio = 0
    else:
        special_chars = sum(not c.isalnum() and not c.isspace() for c in cleaned_text)
        special_char_ratio = special_chars / len(cleaned_text)
    
    # 3. SUSPICIOUS WORDS COUNT
    suspicious_words = [
        'free', 'win', 'winner', 'urgent', 'limited', 'offer', 'deal', 
        'money', 'cash', 'prize', 'gift', 'click', 'here', 'now', 
        'verify', 'account', 'bank', 'credit', 'congratulations',
        'guaranteed', 'act now', 'call now', 'order now', 'million'
    ]
    
    text_lower = cleaned_text.lower() if cleaned_text else ''
    sus_words_count = sum(1 for word in suspicious_words if word in text_lower)
    
    return {
        'number_ratio': number_ratio,
        'special_char_ratio': special_char_ratio,
        'sus_words_count': sus_words_count
    }

def comprehensive_preprocessing(df, dataset_info):
    """
    Main preprocessing function that handles the complete pipeline
    """
    print(f"\n🔄 Starting comprehensive preprocessing for {dataset_info['dataset_type']} dataset")
    print(f"📊 Original shape: {dataset_info['shape']}")
    
    # Step 1: Handle missing values
    df_cleaned = handle_missing_values(df)
    
    # Step 2: Standardize column names
    df_cleaned.columns = [col.lower().strip().replace(' ', '_') for col in df_cleaned.columns]
    
    # Update dataset info with new column names
    if dataset_info['text_col']:
        dataset_info['text_col'] = dataset_info['text_col'].lower().strip().replace(' ', '_')
    if dataset_info['label_col']:
        dataset_info['label_col'] = dataset_info['label_col'].lower().strip().replace(' ', '_')
    
    # Step 3: Text preprocessing (if text data exists)
    if dataset_info['text_col'] and dataset_info['text_col'] in df_cleaned.columns:
        print(f"🔤 Processing text column: '{dataset_info['text_col']}'")
        
        # Clean text and replace the original text column
        original_text_col = dataset_info['text_col']
        df_cleaned[original_text_col] = df_cleaned[original_text_col].apply(normalize_text)
        print(f"✅ Replaced raw text with cleaned text in column: '{original_text_col}'")
        
        # Extract crucial features for spam detection
        print("🎯 Extracting crucial spam detection features...")
        
        features = df_cleaned.apply(
            lambda row: extract_crucial_features(
                row[original_text_col]
            ), 
            axis=1
        )
        
        # Add features as separate columns
        for feature_name in features.iloc[0].keys():
            df_cleaned[feature_name] = features.apply(lambda x: x[feature_name])
        
        # Basic text statistics
        df_cleaned['text_length'] = df_cleaned[original_text_col].str.len()
        df_cleaned['word_count'] = df_cleaned[original_text_col].apply(lambda x: len(x.split()) if x else 0)
    
    # Step 4: Label preprocessing (if labels exist)
    if dataset_info['label_col'] and dataset_info['label_col'] in df_cleaned.columns:
        print(f"🏷️ Standardizing labels in column: '{dataset_info['label_col']}'")
        df_cleaned['label'] = standardize_labels(df_cleaned[dataset_info['label_col']])
        
        # Remove original label column if different name
        if dataset_info['label_col'] != 'label':
            df_cleaned = df_cleaned.drop(columns=[dataset_info['label_col']])
    
    # Step 5: Remove duplicate rows
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    removed_duplicates = initial_rows - len(df_cleaned)
    if removed_duplicates > 0:
        print(f"🗑️ Removed {removed_duplicates} duplicate rows")
    
    print(f"✅ Preprocessing complete!")
    print(f"📊 Final shape: {df_cleaned.shape}")
    print(f"📈 Features created: {len(df_cleaned.columns)} total columns")
    
    return df_cleaned

def process_dataset(file_path, output_dir=None):
    """
    Main function to process any email/text dataset
    
    Args:
        file_path: Path to the raw dataset
        output_dir: Directory to save cleaned dataset (optional)
    
    Returns:
        Cleaned DataFrame ready for ML
    """
    print("🚀 STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'
    
    # Extract filename for output naming
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        # Step 1: Load dataset
        print(f"📁 Loading dataset: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"✅ Dataset loaded successfully!")
        
        # Step 2: Analyze dataset structure
        print(f"\n🔍 ANALYZING DATASET STRUCTURE")
        dataset_info = detect_dataset_structure(df)
        
        print(f"📊 Dataset Type: {dataset_info['dataset_type']}")
        print(f"📊 Shape: {dataset_info['shape']}")
        print(f"📊 Text Column: {dataset_info['text_col']}")
        print(f"📊 Label Column: {dataset_info['label_col']}")
        
        # Step 3: Comprehensive preprocessing
        df_cleaned = comprehensive_preprocessing(df, dataset_info)
        
        # Step 4: Save cleaned dataset
        output_filename = f"cleaned_{filename}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        df_cleaned.to_csv(output_path, index=False)
        print(f"\n💾 Cleaned dataset saved: {output_path}")
        
        # Step 5: Generate preprocessing report
        # save_preprocessing_report(df, df_cleaned, output_dir, filename)
        
        # Step 6: Display summary
        print(f"\n📈 PREPROCESSING SUMMARY")
        print("-" * 40)
        print(f"✅ Original rows: {len(df):,}")
        print(f"✅ Cleaned rows: {len(df_cleaned):,}")
        print(f"✅ Features created: {len(df_cleaned.columns)}")
        
        if 'label' in df_cleaned.columns:
            spam_count = df_cleaned['label'].sum()
            ham_count = len(df_cleaned) - spam_count
            print(f"✅ Spam samples: {spam_count:,} ({spam_count/len(df_cleaned):.1%})")
            print(f"✅ Ham samples: {ham_count:,}")
        
        # Feature importance for ML readiness
        crucial_features = ['number_ratio', 'special_char_ratio', 'sus_words_count']
        available_features = [f for f in crucial_features if f in df_cleaned.columns]
        
        if available_features:
            print(f"✅ Crucial ML features: {', '.join(available_features)}")
        
        print(f"\n🎉 PREPROCESSING COMPLETE! Dataset ready for ML training.")
        print("=" * 60)
        
        return df_cleaned
        
    except Exception as e:
        print(f"❌ Error processing dataset: {str(e)}")
        print(f"💡 Please check the file format and structure")
        return None

# Helper function to get feature columns for ML pipeline
def get_feature_columns():
    """Return the list of crucial feature column names for ML pipeline"""
    return ['number_ratio', 'special_char_ratio', 'sus_words_count']

# Example usage and testing
if __name__ == "__main__":
    # Test with different datasets
    test_files = [
        'emails.csv',
        'mail_data.csv',
        # 'combined_data.csv'
        'CEAS_08.csv'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🧪 Testing with: {test_file}")
            cleaned_df = process_dataset(test_file)
            
            if cleaned_df is not None:
                print(f"✅ Successfully processed {test_file}")
                print(f"📋 Sample of cleaned data:")
                print(cleaned_df.head())
            else:
                print(f"❌ Failed to process {test_file}")
        else:
            print(f"⚠️ File not found: {test_file}")