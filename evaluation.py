import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    accuracy_score, roc_curve, precision_recall_curve, auc
)
import joblib
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

MODELS_DIR = Path("saved_models")
PLOTS_DIR = Path("evaluation_plots")

def setup_plots_directory():
    """Create directory for evaluation plots"""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"📁 Plots will be saved to: {PLOTS_DIR}")

def load_saved_model(model_path=None):
    """Load a saved model"""
    if model_path is None:
        model_path = MODELS_DIR / "logistic_regression_spam_classifier_latest.joblib"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    model_artifacts = joblib.load(model_path)
    print(f"📥 Model loaded from: {model_path}")
    print(f"   Training date: {model_artifacts.get('training_date', 'Unknown')}")
    
    return model_artifacts

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    print("\n📊 EVALUATING MODEL PERFORMANCE")
    print("="*50)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"📈 Accuracy: {accuracy:.4f}")
    print(f"📈 AUC-ROC:  {auc_score:.4f}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    print(f"\n📋 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   True Negatives (Ham):  {cm[0,0]:,}")
    print(f"   False Positives:       {cm[0,1]:,}")
    print(f"   False Negatives:       {cm[1,0]:,}")
    print(f"   True Positives (Spam): {cm[1,1]:,}")
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved: {save_path}")
    
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC curve saved: {save_path}")
    
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba, save_path=None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 PR curve saved: {save_path}")
    
    plt.show()

def plot_feature_importance(model, vectorizer, feature_names, top_n=20, save_path=None):
    """Plot top feature importance"""
    if hasattr(model, 'coef_'):
        # Get TF-IDF feature names
        tfidf_features = vectorizer.get_feature_names_out()
        all_features = list(tfidf_features) + feature_names
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(model.coef_[0])
        
        # Create feature importance dataframe
        feature_df = pd.DataFrame({
            'feature': all_features,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Feature Importance (|Coefficient|)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance saved: {save_path}")
        
        plt.show()
    else:
        print("⚠️ Model doesn't have feature coefficients")

def create_evaluation_report(results, save_path=None):
    """Create summary evaluation report"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Evaluation Report', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. ROC Curve (placeholder - would need y_test and y_pred_proba)
    ax2.text(0.5, 0.5, f"AUC: {results['auc']:.4f}\nAccuracy: {results['accuracy']:.4f}", 
             ha='center', va='center', transform=ax2.transAxes, fontsize=14)
    ax2.set_title('Model Metrics')
    
    # 3. Prediction distribution
    y_pred_proba = results['y_pred_proba']
    ax3.hist(y_pred_proba, bins=20, alpha=0.7, color='skyblue')
    ax3.set_xlabel('Prediction Probability')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Prediction Probability Distribution')
    
    # 4. Performance summary
    metrics_text = f"""
    Accuracy: {results['accuracy']:.4f}
    AUC-ROC: {results['auc']:.4f}
    
    True Negatives: {cm[0,0]}
    False Positives: {cm[0,1]}
    False Negatives: {cm[1,0]}
    True Positives: {cm[1,1]}
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12, 
             verticalalignment='center')
    ax4.set_title('Performance Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation report saved: {save_path}")
    
    plt.show()

def main():
    """Main evaluation pipeline"""
    print("="*50)
    print("📊 SPAM CLASSIFIER EVALUATION")
    print("="*50)
    
    try:
        setup_plots_directory()
        
        # Load saved model
        model_artifacts = load_saved_model()
        if not model_artifacts:
            print("❌ No model found for evaluation")
            return
        
        model = model_artifacts['model']
        vectorizer = model_artifacts['vectorizer']
        feature_names = model_artifacts['feature_names']
        
        # For evaluation, we need test data - you might need to modify this
        # based on how you want to get your test data
        print("⚠️ Note: You need to provide test data (X_test, y_test) for evaluation")
        print("   This can be done by modifying the training script to save test data")
        print("   or by loading and splitting your data again")

        # Example: Load test data from CSV and transform using vectorizer
        # Replace 'test_data.csv' and 'label_column' with your actual file and label column
        test_data_path = Path("test_data.csv")
        if not test_data_path.exists():
            print(f"❌ Test data file not found: {test_data_path}")
            return

        df_test = pd.read_csv(test_data_path)
        y_test = df_test['label_column']  # Change 'label_column' to your actual label column name
        X_test_raw = df_test.drop(columns=['label_column'])  # Drop label column to get features
        # If your features are text, use vectorizer to transform
        if hasattr(vectorizer, "transform"):
            X_test = vectorizer.transform(X_test_raw.squeeze())
        else:
            X_test = X_test_raw.values

        results = evaluate_model(model, X_test, y_test)

        # Create visualizations
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

        plot_confusion_matrix(results['confusion_matrix'], 
                             PLOTS_DIR / f'confusion_matrix_{timestamp}.png')

        plot_roc_curve(y_test, results['y_pred_proba'],
                      PLOTS_DIR / f'roc_curve_{timestamp}.png')

        plot_precision_recall_curve(y_test, results['y_pred_proba'],
                                  PLOTS_DIR / f'pr_curve_{timestamp}.png')

        plot_feature_importance(model, vectorizer, feature_names,
                              save_path=PLOTS_DIR / f'feature_importance_{timestamp}.png')

        create_evaluation_report(results,
                               save_path=PLOTS_DIR / f'evaluation_report_{timestamp}.png')

        print(f"\n🎉 Evaluation setup completed!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()