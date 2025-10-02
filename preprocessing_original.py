# import pandas as pd
# import re
# import numpy as np

# def preprocess_email_data(df):
#     """Preprocess email dataset with various cleaning and standardization steps"""
    
#     # Standardize column names
#     df.columns = [col.lower().strip() for col in df.columns]
    
#     # Handle different text column names
#     text_col = None
#     for col in ['text', 'text_combined', 'body', 'message', 'email']:
#         if col in df.columns:
#             text_col = col
#             break
    
#     if text_col is None:
#         raise ValueError("No text column found in dataset")

#     # Handle different label column names and formats
#     label_col = None
#     for col in ['label', 'class', 'spam', 'category']:
#         if col in df.columns:
#             label_col = col
#             break
    
#     if label_col is None:
#         raise ValueError("No label column found in dataset")

#     # Clean text data
#     df[text_col] = df[text_col].astype(str)
    
#     # Extract subject if present in text
#     df['subject'] = df[text_col].apply(extract_subject)
#     df[text_col] = df[text_col].apply(remove_subject_prefix)
    
#     # Clean text content
#     df['clean_text'] = df[text_col].apply(clean_text)
    
#     # Standardize labels to 0/1
#     df[label_col] = standardize_labels(df[label_col])
    
#     # Feature engineering
#     df['text_length'] = df['clean_text'].apply(len)
#     df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
#     df['avg_word_length'] = df['clean_text'].apply(average_word_length)
#     df['number_ratio'] = df['clean_text'].apply(number_ratio)
#     df['special_char_ratio'] = df['clean_text'].apply(special_char_ratio)
#     df['uppercase_ratio'] = df['clean_text'].apply(uppercase_ratio)
    
#     return df

# def extract_subject(text):
#     """Extract subject from email text if present"""
#     subject_match = re.search(r'Subject: (.*?)(?:\n|$)', text, re.IGNORECASE)
#     if subject_match:
#         return subject_match.group(1).strip()
#     return ''

# def remove_subject_prefix(text):
#     """Remove 'Subject:' prefix from text"""
#     return re.sub(r'Subject: .*?\n', '', text, flags=re.IGNORECASE)

# def clean_text(text):
#     """Clean text content while preserving important punctuation"""
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove HTML tags
#     text = re.sub(r'<[^>]+>', ' ', text)
    
#     # Keep important punctuation but remove other special characters
#     text = re.sub(r'[^a-z0-9\s.,!?\'"-]', ' ', text)
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text

# def standardize_labels(labels):
#     """Convert various label formats to binary 0/1"""
#     # Convert to string and lowercase
#     labels = labels.astype(str).str.lower()
    
#     # Map common spam indicators to 1
#     spam_indicators = ['1', 'spam', 'true', 'yes']
#     return labels.apply(lambda x: 1 if x in spam_indicators else 0)

# def average_word_length(text):
#     """Calculate average word length"""
#     words = text.split()
#     if not words:
#         return 0
#     return sum(len(word) for word in words) / len(words)

# def number_ratio(text):
#     """Calculate ratio of numbers to total length"""
#     if not text:
#         return 0
#     return sum(c.isdigit() for c in text) / len(text)

# def special_char_ratio(text):
#     """Calculate ratio of special characters to total length"""
#     if not text:
#         return 0
#     special_chars = sum(not c.isalnum() and not c.isspace() for c in text)
#     return special_chars / len(text)

# def uppercase_ratio(text):
#     """Calculate ratio of uppercase letters in original text"""
#     if not text:
#         return 0
#     return sum(c.isupper() for c in text) / len(text)

# def process_dataset(file_path):
#     """Main function to process email dataset"""
#     # Read dataset
#     df = pd.read_csv(file_path, low_memory=False)
    
#     # Apply preprocessing
#     cleaned_df = preprocess_email_data(df)
    
#     # # Save processed dataset
#     # output_path = file_path.replace('.csv', '_processed.csv')
#     # cleaned_df.to_csv(output_path, index=False)

#     # Generate output path by inserting 'cleaned_' before the filename
#     # import os
#     # base_dir = os.path.dirname(file_path)
#     # filename = os.path.basename(file_path)
#     # output_filename = f"cleaned_{filename}"
#     # output_path = os.path.join(base_dir, output_filename)
    
#     # # Save processed dataset
#     cleaned_df.to_csv("cleaned_spam_email_dataset.csv", index=False)
#     print(f"Cleaned dataset saved to: {"cleaned_spam_email_dataset.csv"}")
    
#     return cleaned_df





# import pandas as pd
# import re
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import CountVectorizer

# def normalize_repeated_symbols(text):
#     """Normalize repeated grammatical symbols to appear only once"""
#     symbols = {'.': 0, ',': 0, '!': 0, '?': 0, '-': 0, '"': 0, "'": 0}
    
#     for char in text:
#         if char in symbols:
#             symbols[char] += 1
    
#     for symbol in symbols:
#         if symbols[symbol] >= 2:
#             pattern = f'\\{symbol}+'
#             text = re.sub(pattern, symbol, text)
    
#     return text, symbols

# def clean_text(text):
#     """Clean text content while preserving important punctuation"""
#     text = text.lower()
#     text = re.sub(r'<[^>]+>', ' ', text)
#     text = re.sub(r'[^a-z0-9\s.,!?\'"-]', ' ', text)
#     text, _ = normalize_repeated_symbols(text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def extract_subject(text):
#     """Extract subject from email text if present"""
#     subject_match = re.search(r'Subject: (.*?)(?:\n|$)', text, re.IGNORECASE)
#     if subject_match:
#         return subject_match.group(1).strip()
#     return ''

# def remove_subject_prefix(text):
#     """Remove 'Subject:' prefix from text"""
#     return re.sub(r'Subject: .*?\n', '', text, flags=re.IGNORECASE)

# def detect_column_types(df):
#     """Automatically detect text and label columns"""
#     text_col = None
#     label_col = None
    
#     # Common text column names
#     text_candidates = ['text', 'body', 'message', 'email', 'content', 'mail']
#     for col in df.columns:
#         col_lower = col.lower().strip()
#         if col_lower in text_candidates:
#             text_col = col
#             break
    
#     # If no exact match, look for columns with long text content
#     if text_col is None:
#         for col in df.columns:
#             if df[col].dtype == 'object':
#                 avg_length = df[col].astype(str).str.len().mean()
#                 if avg_length > 50:
#                     text_col = col
#                     break
    
#     # Common label column names
#     label_candidates = ['label', 'class', 'spam', 'category', 'target']
#     for col in df.columns:
#         col_lower = col.lower().strip()
#         if col_lower in label_candidates:
#             label_col = col
#             break
    
#     return text_col, label_col

# def standardize_labels(labels):
#     """Convert various label formats to binary 0/1"""
#     labels = labels.astype(str).str.lower().str.strip()
    
#     # Map spam indicators to 1, everything else to 0
#     spam_indicators = ['1', 'spam', 'true', 'yes', '1.0']
#     ham_indicators = ['0', 'ham', 'false', 'no', '0.0']
    
#     def convert_label(x):
#         if x in spam_indicators:
#             return 1
#         elif x in ham_indicators:
#             return 0
#         else:
#             try:
#                 val = float(x)
#                 return 1 if val > 0.5 else 0
#             except:
#                 return 0
    
#     return labels.apply(convert_label)

# def extract_spam_features(text):
#     """Extract spam-specific features from text"""
#     susp_words = [
#         'free', 'win', 'urgent', 'bank', 'prize', 'verify', 'account', 'gift',
#         'money', 'cash', 'deal', 'offer', 'limited', 'act now', 'click here',
#         'congratulations', 'winner', 'credit', 'loan', 'debt', 'million'
#     ]
    
#     text_lower = text.lower()
    
#     # Count suspicious words
#     susp_count = sum(1 for word in susp_words if word in text_lower)
#     has_susp = int(susp_count > 0)
    
#     # Count links
#     num_links = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    
#     # Count email addresses
#     num_emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    
#     # Count phone numbers
#     num_phones = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
    
#     # Count exclamation marks and question marks
#     num_exclamations = text.count('!')
#     num_questions = text.count('?')
    
#     # Count ALL CAPS words
#     words = text.split()
#     caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
#     caps_ratio = caps_words / max(len(words), 1)
    
#     return {
#         'susp_word_count': susp_count,
#         'has_suspicious': has_susp,
#         'num_links': num_links,
#         'num_emails': num_emails,
#         'num_phones': num_phones,
#         'num_exclamations': num_exclamations,
#         'num_questions': num_questions,
#         'caps_ratio': caps_ratio
#     }

# def basic_text_features(text):
#     """Extract basic text statistics"""
#     return {
#         'text_length': len(text),
#         'word_count': len(text.split()),
#         'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
#         'number_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
#         'special_char_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1),
#         'uppercase_ratio': sum(c.isupper() for c in text) / max(len(text), 1)
#     }

# def preprocess_email_data(df):
#     """Comprehensive email preprocessing with automatic column detection"""
    
#     # Standardize column names
#     df.columns = [col.lower().strip() for col in df.columns]
    
#     # Automatically detect text and label columns
#     text_col, label_col = detect_column_types(df)
    
#     if text_col is None:
#         raise ValueError(f"Could not find text column. Available columns: {list(df.columns)}")
#     if label_col is None:
#         raise ValueError(f"Could not find label column. Available columns: {list(df.columns)}")
    
#     print(f"Detected text column: '{text_col}'")
#     print(f"Detected label column: '{label_col}'")
    
#     # Clean and prepare data
#     df = df.dropna(subset=[text_col, label_col])
#     df[text_col] = df[text_col].astype(str)
    
#     # Extract subject if present
#     df['subject'] = df[text_col].apply(extract_subject)
#     df[text_col] = df[text_col].apply(remove_subject_prefix)
    
#     # Clean text
#     df['clean_text'] = df[text_col].apply(clean_text)
    
#     # Standardize labels
#     df['label'] = standardize_labels(df[label_col])
    
#     # Extract basic text features
#     basic_features = df['clean_text'].apply(basic_text_features)
#     for feature_name in basic_features.iloc[0].keys():
#         df[feature_name] = basic_features.apply(lambda x: x[feature_name])
    
#     # Extract spam-specific features
#     spam_features = df[text_col].apply(extract_spam_features)
#     for feature_name in spam_features.iloc[0].keys():
#         df[feature_name] = spam_features.apply(lambda x: x[feature_name])
    
#     # Add symbol counts
#     text_symbols = df[text_col].apply(lambda x: normalize_repeated_symbols(x)[1])
#     for symbol in ['.', ',', '!', '?', '-', '"', "'"]:
#         df[f'{symbol}_count'] = text_symbols.apply(lambda x: x[symbol])
    
#     return df

# def process_dataset(file_path):
#     """Main function to process email dataset"""
#     # Read dataset
#     print(f"Loading dataset from: {file_path}")
#     df = pd.read_csv(file_path, low_memory=False)
#     print(f"Dataset shape: {df.shape}")
    
#     # Apply preprocessing
#     cleaned_df = preprocess_email_data(df)
    
#     # Generate output path
#     import os
#     base_dir = os.path.dirname(file_path)
#     filename = os.path.basename(file_path)
#     output_filename = f"cleaned_{filename}"
#     output_path = os.path.join(base_dir, output_filename)
    
#     # Save processed dataset
#     cleaned_df.to_csv(output_path, index=False)
#     print(f"Cleaned dataset saved to: {output_path}")
#     print(f"Features added: {len(cleaned_df.columns)} total columns")
    
#     return cleaned_df



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
        # 'CEAS_08.csv'
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
    from datasets import load_dataset
    
    ds = load_dataset("yxzwayne/email-spam-10k", split="train")
    
    # Test with different datasets
    test_files = [
        'emails.csv',
        'mail_data.csv',
        'CEAS_08.csv',
        ds  # HuggingFace dataset
    ]
    
    for test_file in test_files:
        # Handle HuggingFace datasets
        if hasattr(test_file, 'to_pandas'):  # Check if it's a HF dataset
            print(f"\n🧪 Testing with HuggingFace dataset: {type(test_file).__name__}")
            try:
                # Convert HF dataset to pandas DataFrame
                df = test_file.to_pandas()
                
                # Process the DataFrame directly
                dataset_info = detect_dataset_structure(df)
                cleaned_df = comprehensive_preprocessing(df, dataset_info)
                
                if cleaned_df is not None:
                    print(f"✅ Successfully processed HuggingFace dataset")
                    print(f"📋 Sample of cleaned data:")
                    print(cleaned_df.head())
                else:
                    print(f"❌ Failed to process HuggingFace dataset")
                    
            except Exception as e:
                print(f"❌ Error processing HuggingFace dataset: {e}")
                
        # Handle regular CSV files
        elif isinstance(test_file, str) and os.path.exists(test_file):
            print(f"\n🧪 Testing with: {test_file}")
            cleaned_df = process_dataset(test_file)
            
            if cleaned_df is not None:
                print(f"✅ Successfully processed {test_file}")
                print(f"📋 Sample of cleaned data:")
                print(cleaned_df.head())
            else:
                print(f"❌ Failed to process {test_file}")
                
        elif isinstance(test_file, str):
            print(f"⚠️ File not found: {test_file}")
        else:
            print(f"⚠️ Unknown data type: {type(test_file)}")