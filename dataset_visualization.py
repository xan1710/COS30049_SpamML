import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('datasets/cleaned_datasets/combined_email_dataset.csv')

# Create a focused visualization dashboard with 4 crucial plots
fig = plt.figure(figsize=(10, 8))

# 1. Class Distribution - Understanding data balance
plt.subplot(2, 2, 1)
class_counts = df['label'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.pie(class_counts.to_numpy(), labels=['Ham', 'Spam'], autopct='%1.1f%%', colors=colors)
plt.title('Email Classification Distribution', fontsize=14, fontweight='bold')

# 2. Feature Correlation Matrix - Understanding feature relationships
plt.subplot(2, 2, 2)
numeric_cols = ['label', 'number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# 3. Key Feature Distributions - Most discriminative features
plt.subplot(2, 2, 3)
plt.hist(df[df['label'] == 0]['spam_words'], bins=15, alpha=0.7, label='Ham', color='#2ecc71', density=True)
plt.hist(df[df['label'] == 1]['spam_words'], bins=15, alpha=0.7, label='Spam', color='#e74c3c', density=True)
plt.xlabel('Spam Words Count')
plt.ylabel('Density')
plt.title('Spam Words Distribution by Class', fontsize=14, fontweight='bold')
plt.legend()

# 4. Feature Comparison by Class - Overview of all features
plt.subplot(2, 2, 4)
feature_means = df.groupby('label')[['number_ratio', 'special_char_ratio', 'spam_words', 'text_length']].mean()
x = np.arange(len(feature_means.columns))
width = 0.35

plt.bar(x - width/2, feature_means.loc[0], width, label='Ham', color='#2ecc71', alpha=0.8)
plt.bar(x + width/2, feature_means.loc[1], width, label='Spam', color='#e74c3c', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Average Value')
plt.title('Average Feature Values by Class', fontsize=14, fontweight='bold')
plt.xticks(x, ['Number\nRatio', 'Special Char\nRatio', 'Spam\nWords', 'Text\nLength'])
plt.legend()

plt.suptitle('Email Spam Detection - Key Dataset Insights', fontsize=16, fontweight='bold', y=0.98)
plt.show()

# Print essential dataset overview
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total emails: {len(df)}")
print(f"Ham emails: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)")
print(f"Spam emails: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("KEY FEATURE STATISTICS")
print("=" * 60)
key_features = ['spam_words', 'special_char_ratio', 'number_ratio', 'text_length']
print(df.groupby('label')[key_features].describe().round(3))