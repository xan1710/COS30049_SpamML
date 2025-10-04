import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)
from preprocessing import load_dataset, extract_features
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    """Prepare features for both models"""
    X_text = df['text'].fillna('')
    y = df['label']
    
    # TF-IDF features for Logistic Regression
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = vectorizer.fit_transform(X_text)
    
    # Numeric features for KNN
    feature_columns = ['number_ratio', 'special_char_ratio', 'spam_words', 'text_length', 'word_count']
    X_numeric = df[feature_columns]
    
    return X_tfidf, X_numeric, y, vectorizer

def train_models(X_tfidf, X_numeric, y):
    """Train both models"""
    # Split data
    X_tfidf_train, X_tfidf_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    X_num_train, X_num_test, _, _ = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numeric features for KNN
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    
    # Train Logistic Regression
    log_reg = LogisticRegression(C=10, random_state=42, max_iter=20000)
    log_reg.fit(X_tfidf_train, y_train)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_num_train_scaled, y_train)
    
    return {
        'models': {'Logistic Regression': log_reg, 'KNN': knn},
        'test_data': {
            'X_tfidf_test': X_tfidf_test,
            'X_num_test_scaled': X_num_test_scaled,
            'y_test': y_test
        },
        'scaler': scaler
    }

def evaluate_models(models_data):
    """Evaluate both models and return metrics"""
    models = models_data['models']
    test_data = models_data['test_data']
    
    results = {}
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            X_test = test_data['X_tfidf_test']
        else:
            X_test = test_data['X_num_test_scaled']
        
        y_test = test_data['y_test']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    return results

def plot_comparison(results):
    """Create comprehensive comparison plots"""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison: Logistic Regression vs KNN', fontsize=16, fontweight='bold')
    
    # 1. Metrics Comparison Bar Plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    models = list(results.keys())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    log_reg_scores = [results['Logistic Regression'][metric] for metric in metrics]
    knn_scores = [results['KNN'][metric] for metric in metrics]
    
    axes[0, 0].bar(x - width/2, log_reg_scores, width, label='Logistic Regression', alpha=0.8)
    axes[0, 0].bar(x + width/2, knn_scores, width, label='KNN', alpha=0.8)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confusion Matrices
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                   ax=axes[0, i+1])
        axes[0, i+1].set_title(f'{name} - Confusion Matrix')
    
    # 3. ROC Curves
    axes[1, 0].set_title('ROC Curves Comparison')
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
        auc_score = result['auc']
        axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Prediction Probability Distribution
    for i, (name, result) in enumerate(results.items()):
        spam_probs = result['y_proba'][result['y_test'] == 1]
        ham_probs = result['y_proba'][result['y_test'] == 0]
        
        axes[1, i+1].hist(ham_probs, bins=20, alpha=0.7, label='Ham', density=True)
        axes[1, i+1].hist(spam_probs, bins=20, alpha=0.7, label='Spam', density=True)
        axes[1, i+1].set_xlabel('Prediction Probability')
        axes[1, i+1].set_ylabel('Density')
        axes[1, i+1].set_title(f'{name} - Probability Distribution')
        axes[1, i+1].legend()
        axes[1, i+1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_detailed_comparison(results):
    """Print detailed comparison results"""
    print("="*60)
    print("DETAILED MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        metric: [results[model][metric] for model in results.keys()]
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
    }, index=list(results.keys()))
    
    print("\n📊 Performance Metrics Summary:")
    print(comparison_df.round(4))
    
    # Determine best model for each metric
    print("\n🏆 Best Performance by Metric:")
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"  {metric.capitalize()}: {best_model} ({best_score:.4f})")
    
    # Overall winner
    overall_scores = comparison_df.mean(axis=1)
    overall_winner = overall_scores.idxmax()
    print(f"\n🎯 Overall Best Model: {overall_winner} (Average Score: {overall_scores.max():.4f})")
    
    # Print classification reports
    for name, result in results.items():
        print(f"\n📋 {name} - Classification Report:")
        print(classification_report(result['y_test'], result['y_pred'], 
                                  target_names=['Ham', 'Spam']))

def main():
    """Main execution function"""
    print("🚀 Loading dataset and preparing features...")
    
    # Load dataset
    df = load_dataset()
    if df is None:
        raise ValueError("No data found. Run preprocessing first.")
    
    print(f"📊 Dataset: {len(df):,} samples, {df['label'].mean():.1%} spam")
    
    # Prepare features
    X_tfidf, X_numeric, y, vectorizer = prepare_features(df)
    
    print("🎯 Training models...")
    # Train models
    models_data = train_models(X_tfidf, X_numeric, y)
    
    print("📈 Evaluating models...")
    # Evaluate models
    results = evaluate_models(models_data)
    
    # Print comparison
    print_detailed_comparison(results)
    
    # Plot comparison
    print("📊 Generating visualization...")
    plot_comparison(results)
    
    print("\n✅ Model comparison completed!")

if __name__ == "__main__":
    main()