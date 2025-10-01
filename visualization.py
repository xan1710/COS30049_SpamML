import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import joblib
import os
from datetime import datetime

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpamClassificationVisualizer:
    def __init__(self, save_dir='.'):
        """Initialize visualizer with save directory"""
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualizations directory
        self.viz_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        """Save plot with timestamp"""
        filepath = os.path.join(self.viz_dir, f"{filename}_{self.timestamp}.png")
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"📊 Saved: {filepath}")
        return filepath
    
    def plot_dataset_overview(self, df):
        """Create overview plots of the dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Class distribution
        label_counts = df['label'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0, 0].pie(label_counts.values, labels=['Ham', 'Spam'], 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Text length distribution by class
        if 'text_length' in df.columns:
            sns.histplot(data=df, x='text_length', hue='label', bins=50, 
                        ax=axes[0, 1], alpha=0.7)
            axes[0, 1].set_title('Text Length Distribution by Class')
            axes[0, 1].set_xlabel('Text Length (characters)')
        
        # 3. Word count distribution by class
        if 'word_count' in df.columns:
            sns.boxplot(data=df, x='label', y='word_count', ax=axes[1, 0])
            axes[1, 0].set_title('Word Count Distribution by Class')
            axes[1, 0].set_xticklabels(['Ham', 'Spam'])
        
        # 4. Feature correlation heatmap
        feature_cols = ['number_ratio', 'special_char_ratio', 'sus_words_count', 
                       'text_length', 'word_count', 'label']
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) > 2:
            corr_matrix = df[available_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[1, 1], square=True)
            axes[1, 1].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        return self.save_plot('dataset_overview')
    
    def plot_feature_analysis(self, df):
        """Analyze key features for spam detection"""
        feature_cols = ['number_ratio', 'special_char_ratio', 'sus_words_count']
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            print("⚠️ No engineered features found for analysis")
            return None
        
        n_features = len(available_features)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 6))
        if n_features == 1:
            axes = [axes]
        
        fig.suptitle('Key Features Analysis for Spam Detection', fontsize=16, fontweight='bold')
        
        feature_titles = {
            'number_ratio': 'Number Ratio Distribution',
            'special_char_ratio': 'Special Character Ratio',
            'sus_words_count': 'Suspicious Words Count'
        }
        
        for i, feature in enumerate(available_features):
            # Box plot for each feature by class
            sns.boxplot(data=df, x='label', y=feature, ax=axes[i])
            axes[i].set_title(feature_titles.get(feature, feature))
            axes[i].set_xticklabels(['Ham', 'Spam'])
            
            # Add mean values as text
            ham_mean = df[df['label'] == 0][feature].mean()
            spam_mean = df[df['label'] == 1][feature].mean()
            axes[i].text(0, axes[i].get_ylim()[1]*0.9, f'Mean: {ham_mean:.3f}', 
                        ha='center', fontweight='bold')
            axes[i].text(1, axes[i].get_ylim()[1]*0.9, f'Mean: {spam_mean:.3f}', 
                        ha='center', fontweight='bold')
        
        plt.tight_layout()
        return self.save_plot('feature_analysis')
    
    def plot_model_performance(self, model, X_test, y_test):
        """Plot model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Add accuracy to confusion matrix
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        axes[0, 0].text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
                       ha='center', transform=axes[0, 0].transAxes, fontweight='bold')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        axes[1, 0].plot(recall, precision, color='green', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend(loc="lower left")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Probability Distribution
        spam_probs = y_pred_proba[y_test == 1]
        ham_probs = y_pred_proba[y_test == 0]
        
        axes[1, 1].hist(ham_probs, bins=30, alpha=0.7, label='Ham', color='green', density=True)
        axes[1, 1].hist(spam_probs, bins=30, alpha=0.7, label='Spam', color='red', density=True)
        axes[1, 1].set_xlabel('Spam Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Threshold')
        
        plt.tight_layout()
        return self.save_plot('model_performance')
    
    def plot_learning_curves(self, model, X, y):
        """Plot learning curves to analyze model performance vs dataset size"""
        print("📈 Generating learning curves...")
        
        train_sizes, train_scores, test_scores, _, _ = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(test_scores, axis=1)
        val_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('ROC AUC Score')
        plt.title('Learning Curves - Model Performance vs Dataset Size')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        return self.save_plot('learning_curves')
    
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """Plot feature importance from logistic regression coefficients"""
        if hasattr(model, 'coef_'):
            # Get feature importance (absolute coefficients)
            importance = np.abs(model.coef_[0])
            
            # Create feature importance dataframe
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_df, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} Most Important Features')
            plt.xlabel('Feature Importance (|Coefficient|)')
            plt.tight_layout()
            
            return self.save_plot('feature_importance')
        else:
            print("⚠️ Model doesn't have feature coefficients")
            return None
    
    def plot_prediction_examples(self, model_artifacts, test_emails):
        """Visualize prediction examples"""
        from logistic_regression_pipeline import predict_single_email
        
        # Get predictions for test emails
        results = []
        for email in test_emails:
            result = predict_single_email(email, model_artifacts)
            if 'error' not in result:
                results.append({
                    'email': email[:50] + '...' if len(email) > 50 else email,
                    'prediction': result['prediction'],
                    'spam_probability': result['spam_probability'],
                    'confidence': result['confidence']
                })
        
        if not results:
            print("⚠️ No valid predictions to visualize")
            return None
        
        df_results = pd.DataFrame(results)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Spam probability for each email
        colors = ['red' if pred == 'Spam' else 'green' for pred in df_results['prediction']]
        bars = ax1.barh(range(len(df_results)), df_results['spam_probability'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(df_results)))
        ax1.set_yticklabels(df_results['email'])
        ax1.set_xlabel('Spam Probability')
        ax1.set_title('Spam Probability for Test Emails')
        ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Threshold')
        ax1.legend()
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, df_results['spam_probability'])):
            ax1.text(prob + 0.01, i, f'{prob:.3f}', va='center')
        
        # 2. Confidence distribution
        spam_conf = df_results[df_results['prediction'] == 'Spam']['confidence']
        ham_conf = df_results[df_results['prediction'] == 'Ham']['confidence']
        
        ax2.hist(ham_conf, bins=10, alpha=0.7, label='Ham', color='green', density=True)
        ax2.hist(spam_conf, bins=10, alpha=0.7, label='Spam', color='red', density=True)
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Prediction Confidence Distribution')
        ax2.legend()
        
        plt.tight_layout()
        return self.save_plot('prediction_examples')
    
    def create_summary_report(self, df, model_performance):
        """Create a summary visualization with key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Spam Classification - Summary Report', fontsize=20, fontweight='bold')
        
        # 1. Dataset statistics
        total_samples = len(df)
        spam_samples = df['label'].sum()
        ham_samples = total_samples - spam_samples
        
        stats_data = [total_samples, spam_samples, ham_samples]
        stats_labels = ['Total\nSamples', 'Spam\nSamples', 'Ham\nSamples']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = axes[0, 0].bar(stats_labels, stats_data, color=colors, alpha=0.8)
        axes[0, 0].set_title('Dataset Statistics')
        axes[0, 0].set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_data):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(stats_data),
                           f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Class balance pie chart
        axes[0, 1].pie([ham_samples, spam_samples], labels=['Ham', 'Spam'], 
                      autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0, 1].set_title('Class Balance')
        
        # 3. Feature statistics
        if all(col in df.columns for col in ['number_ratio', 'special_char_ratio', 'sus_words_count']):
            feature_means = df.groupby('label')[['number_ratio', 'special_char_ratio', 'sus_words_count']].mean()
            
            x = np.arange(len(feature_means.columns))
            width = 0.35
            
            axes[0, 2].bar(x - width/2, feature_means.loc[0], width, label='Ham', color='#2ecc71', alpha=0.8)
            axes[0, 2].bar(x + width/2, feature_means.loc[1], width, label='Spam', color='#e74c3c', alpha=0.8)
            
            axes[0, 2].set_xlabel('Features')
            axes[0, 2].set_ylabel('Average Value')
            axes[0, 2].set_title('Feature Comparison: Ham vs Spam')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(['Number\nRatio', 'Special Char\nRatio', 'Suspicious\nWords'])
            axes[0, 2].legend()
        
        # 4. Performance metrics (if available)
        if model_performance:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            values = [
                model_performance.get('accuracy', 0),
                model_performance.get('precision', 0),
                model_performance.get('recall', 0),
                model_performance.get('f1_score', 0),
                model_performance.get('roc_auc', 0)
            ]
            
            bars = axes[1, 0].bar(metrics, values, color='#9b59b6', alpha=0.8)
            axes[1, 0].set_title('Model Performance Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Text length comparison
        if 'text_length' in df.columns:
            sns.violinplot(data=df, x='label', y='text_length', ax=axes[1, 1])
            axes[1, 1].set_title('Text Length Distribution')
            axes[1, 1].set_xticklabels(['Ham', 'Spam'])
        
        # 6. Word count comparison  
        if 'word_count' in df.columns:
            sns.violinplot(data=df, x='label', y='word_count', ax=axes[1, 2])
            axes[1, 2].set_title('Word Count Distribution')
            axes[1, 2].set_xticklabels(['Ham', 'Spam'])
        
        plt.tight_layout()
        return self.save_plot('summary_report')

def visualize_complete_pipeline(df, model_artifacts, X_test, y_test, test_emails, model_performance=None):
    """Main function to create all visualizations"""
    print("\n🎨 GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 50)
    
    visualizer = SpamClassificationVisualizer()
    
    # Create all visualizations
    plots_created = []
    
    print("📊 Creating dataset overview...")
    plots_created.append(visualizer.plot_dataset_overview(df))
    
    print("📊 Analyzing key features...")
    plots_created.append(visualizer.plot_feature_analysis(df))
    
    print("📊 Evaluating model performance...")
    model, vectorizer, feature_names = model_artifacts
    plots_created.append(visualizer.plot_model_performance(model, X_test, y_test))
    
    print("📊 Creating learning curves...")
    # Note: This might take time for large datasets
    # plots_created.append(visualizer.plot_learning_curves(model, X_test, y_test))
    
    print("📊 Analyzing feature importance...")
    all_features = vectorizer.get_feature_names_out().tolist() + feature_names
    plots_created.append(visualizer.plot_feature_importance(model, all_features))
    
    print("📊 Visualizing prediction examples...")
    plots_created.append(visualizer.plot_prediction_examples(model_artifacts, test_emails))
    
    print("📊 Creating summary report...")
    plots_created.append(visualizer.create_summary_report(df, model_performance))
    
    # Filter out None values
    plots_created = [plot for plot in plots_created if plot is not None]
    
    print(f"\n✅ Created {len(plots_created)} visualization files:")
    for plot_path in plots_created:
        print(f"   📄 {os.path.basename(plot_path)}")
    
    print(f"\n📁 All visualizations saved in: {visualizer.viz_dir}")
    return plots_created

# Example usage function
if __name__ == "__main__":
    # This would typically be called from the main pipeline
    print("🎨 Visualization module loaded successfully!")
    print("Use visualize_complete_pipeline() to generate all plots")