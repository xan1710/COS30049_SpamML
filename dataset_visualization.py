import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('datasets/cleaned_datasets/combined_email_dataset.csv')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 15))

# 1. Class Distribution
plt.subplot(3, 4, 1)
class_counts = df['label'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.pie(class_counts.to_numpy(), labels=['Ham', 'Spam'], autopct='%1.1f%%', colors=colors)
plt.title('Email Classification Distribution', fontsize=14, fontweight='bold')

# 2. Text Length Distribution
plt.subplot(3, 4, 2)
plt.hist(df[df['label'] == 0]['text_length'], bins=30, alpha=0.7, label='Ham', color='#2ecc71')
plt.hist(df[df['label'] == 1]['text_length'], bins=30, alpha=0.7, label='Spam', color='#e74c3c')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Text Length Distribution by Class')
plt.legend()
plt.yscale('log')

# 3. Word Count Distribution
plt.subplot(3, 4, 3)
plt.boxplot([df[df['label'] == 0]['word_count'], df[df['label'] == 1]['word_count']], 
            patch_artist=True, 
            boxprops=dict(facecolor='lightblue'))
plt.xticks([1, 2], ['Ham', 'Spam'])
plt.ylabel('Word Count')
plt.title('Word Count Distribution by Class')

# 4. Number Ratio Distribution
plt.subplot(3, 4, 4)
plt.hist(df[df['label'] == 0]['number_ratio'], bins=20, alpha=0.7, label='Ham', color='#2ecc71')
plt.hist(df[df['label'] == 1]['number_ratio'], bins=20, alpha=0.7, label='Spam', color='#e74c3c')
plt.xlabel('Number Ratio')
plt.ylabel('Frequency')
plt.title('Number Ratio Distribution')
plt.legend()

# 5. Special Character Ratio Distribution
plt.subplot(3, 4, 5)
plt.hist(df[df['label'] == 0]['special_char_ratio'], bins=20, alpha=0.7, label='Ham', color='#2ecc71')
plt.hist(df[df['label'] == 1]['special_char_ratio'], bins=20, alpha=0.7, label='Spam', color='#e74c3c')
plt.xlabel('Special Character Ratio')
plt.ylabel('Frequency')
plt.title('Special Character Ratio Distribution')
plt.legend()

# 6. Spam Words Distribution
plt.subplot(3, 4, 6)
spam_word_counts = df['spam_words'].value_counts().head(6)
plt.bar(range(len(spam_word_counts)), spam_word_counts, color='#3498db')
plt.xlabel('Number of Spam Words')
plt.ylabel('Frequency')
plt.title('Spam Words Count Distribution')
plt.xticks(range(len(spam_word_counts)), [str(x) for x in spam_word_counts.index])

# 7. Correlation Heatmap
plt.subplot(3, 4, 7)
numeric_cols = ['label', 'number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')

# 8. Feature Comparison by Class
plt.subplot(3, 4, 8)
feature_means = df.groupby('label')[['number_ratio', 'special_char_ratio', 'spam_words']].mean()
feature_means.T.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
plt.title('Average Feature Values by Class')
plt.ylabel('Average Value')
plt.legend(['Ham', 'Spam'])
plt.xticks(rotation=45)

# 9. Text Length vs Word Count Scatter
plt.subplot(3, 4, 9)
ham_data = df[df['label'] == 0]
spam_data = df[df['label'] == 1]
plt.scatter(ham_data['text_length'], ham_data['word_count'], alpha=0.6, c='#2ecc71', label='Ham', s=20)
plt.scatter(spam_data['text_length'], spam_data['word_count'], alpha=0.6, c='#e74c3c', label='Spam', s=20)
plt.xlabel('Text Length')
plt.ylabel('Word Count')
plt.title('Text Length vs Word Count')
plt.legend()

# 10. Average Word Length
plt.subplot(3, 4, 10)
df['avg_word_length'] = df['text_length'] / df['word_count']
df['avg_word_length'] = df['avg_word_length'].fillna(0)
plt.hist(df[df['label'] == 0]['avg_word_length'], bins=20, alpha=0.7, label='Ham', color='#2ecc71')
plt.hist(df[df['label'] == 1]['avg_word_length'], bins=20, alpha=0.7, label='Spam', color='#e74c3c')
plt.xlabel('Average Word Length')
plt.ylabel('Frequency')
plt.title('Average Word Length Distribution')
plt.legend()

# 11. Feature Distribution Violin Plot
plt.subplot(3, 4, 11)
df_melted = pd.melt(df[['label', 'number_ratio', 'special_char_ratio']], 
                    id_vars=['label'], var_name='feature', value_name='value')
sns.violinplot(data=df_melted, x='feature', y='value', hue='label', split=True)
plt.title('Feature Distribution Comparison')
plt.xticks(rotation=45)

# 12. Summary Statistics Table
plt.subplot(3, 4, 12)
plt.axis('off')
summary_stats = df.groupby('label')[numeric_cols[1:]].agg(['mean', 'std']).round(3)
table_text = []
for label in [0, 1]:
    class_name = 'Ham' if label == 0 else 'Spam'
    table_text.append([f'{class_name} Class Statistics', '', ''])
    for col in numeric_cols[1:]:
        mean_val = summary_stats.loc[label, (col, 'mean')]
        std_val = summary_stats.loc[label, (col, 'std')]
        table_text.append([col, f'{mean_val:.3f}', f'±{std_val:.3f}'])

plt.table(cellText=table_text, colLabels=['Feature', 'Mean', 'Std'], 
          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
plt.title('Summary Statistics by Class', pad=20)

plt.tight_layout()
plt.suptitle('Email Spam Detection Dataset - Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.98)
plt.show()

# Print dataset overview
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total emails: {len(df)}")
print(f"Ham emails: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)")
print(f"Spam emails: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)")
print(f"Features: {list(df.columns)}")

print("\n" + "=" * 60)
print("FEATURE STATISTICS BY CLASS")
print("=" * 60)
print(df.groupby('label')[numeric_cols[1:]].describe().round(3))