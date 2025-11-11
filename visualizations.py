"""
Visualization and reporting tools for churn analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import joblib
from pathlib import Path
from config import REPORTS_DIR, MODELS_DIR, DATA_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ChurnAnalyzer:
    """Comprehensive churn analysis and visualization"""
    
    def __init__(self, model, feature_engineer, X_test, y_test, X_train=None, y_train=None):
        """
        Initialize analyzer
        
        Args:
            model: Trained churn model
            feature_engineer: Fitted feature engineer
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional)
            y_train: Training labels (optional)
        """
        self.model = model
        self.fe = feature_engineer
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        
        # Generate predictions
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)
    
    def plot_confusion_matrix(self, save=True):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {self.model.model_type.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save:
            plt.savefig(REPORTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {REPORTS_DIR / 'confusion_matrix.png'}")
        plt.close()
    
    def plot_roc_curve(self, save=True):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = self.model.metrics['test']['roc_auc']
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model.model_type.upper()}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(REPORTS_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {REPORTS_DIR / 'roc_curve.png'}")
        plt.close()
    
    def plot_precision_recall_curve(self, save=True):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        avg_precision = self.model.metrics['test']['avg_precision']
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model.model_type.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(REPORTS_DIR / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {REPORTS_DIR / 'precision_recall_curve.png'}")
        plt.close()
    
    def plot_feature_importance(self, top_n=20, save=True):
        """Plot feature importance"""
        if self.model.feature_importance is None:
            print("Feature importance not available for this model")
            return
        
        importance_df = self.model.get_feature_importance(self.fe.feature_names, top_n=top_n)
        
        plt.figure(figsize=(12, 10))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features - {self.model.model_type.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(REPORTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {REPORTS_DIR / 'feature_importance.png'}")
            
            # Also save as CSV
            importance_df.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)
            print(f"Feature importance data saved to {REPORTS_DIR / 'feature_importance.csv'}")
        plt.close()
    
    def plot_probability_distribution(self, save=True):
        """Plot distribution of churn probabilities"""
        plt.figure(figsize=(12, 6))
        
        # Separate by actual class
        churned_probs = self.y_pred_proba[self.y_test == 1]
        retained_probs = self.y_pred_proba[self.y_test == 0]
        
        plt.hist(retained_probs, bins=50, alpha=0.6, label='Retained Customers', color='green')
        plt.hist(churned_probs, bins=50, alpha=0.6, label='Churned Customers', color='red')
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        
        plt.xlabel('Predicted Churn Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Churn Probabilities', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(REPORTS_DIR / 'probability_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Probability distribution saved to {REPORTS_DIR / 'probability_distribution.png'}")
        plt.close()
    
    def plot_threshold_analysis(self, save=True):
        """Analyze model performance across different thresholds"""
        thresholds = np.linspace(0.1, 0.9, 50)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((y_pred_thresh == 1) & (self.y_test == 1))
            fp = np.sum((y_pred_thresh == 1) & (self.y_test == 0))
            fn = np.sum((y_pred_thresh == 0) & (self.y_test == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')
        
        plt.xlabel('Prediction Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance vs. Prediction Threshold', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(REPORTS_DIR / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Threshold analysis saved to {REPORTS_DIR / 'threshold_analysis.png'}")
        plt.close()
    
    def generate_churn_risk_report(self, original_data, save=True):
        """
        Generate detailed churn risk report for all customers
        
        Args:
            original_data: Original customer data with customer_ids
            save: Whether to save the report
            
        Returns:
            DataFrame with churn risk analysis
        """
        # Create risk report
        report = pd.DataFrame({
            'customer_id': original_data['customer_id'].values,
            'churn_probability': self.y_pred_proba,
            'predicted_churn': self.y_pred,
            'actual_churn': self.y_test.values
        })
        
        # Add risk categories
        report['risk_category'] = pd.cut(
            report['churn_probability'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
        
        # Add institution information
        for col in ['institution_type', 'region', 'product_tier', 'revenue_monthly']:
            if col in original_data.columns:
                report[col] = original_data[col].values
        
        # Sort by churn probability
        report = report.sort_values('churn_probability', ascending=False)
        
        if save:
            report.to_csv(REPORTS_DIR / 'churn_risk_report.csv', index=False)
            print(f"Churn risk report saved to {REPORTS_DIR / 'churn_risk_report.csv'}")
            
            # Generate summary statistics
            summary = report.groupby('risk_category').agg({
                'customer_id': 'count',
                'churn_probability': 'mean',
                'revenue_monthly': 'sum' if 'revenue_monthly' in report.columns else 'count'
            }).round(2)
            summary.columns = ['num_customers', 'avg_churn_prob', 'total_revenue']
            
            print("\nChurn Risk Summary:")
            print(summary)
            
            summary.to_csv(REPORTS_DIR / 'churn_risk_summary.csv')
        
        return report
    
    def generate_all_visualizations(self, original_data=None):
        """Generate all visualizations and reports"""
        print("\nGenerating visualizations and reports...")
        print("="*60)
        
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_feature_importance()
        self.plot_probability_distribution()
        self.plot_threshold_analysis()
        
        if original_data is not None:
            self.generate_churn_risk_report(original_data)
        
        print("\n" + "="*60)
        print("All visualizations and reports generated successfully!")
        print(f"Reports saved to: {REPORTS_DIR}")
        print("="*60)


def create_executive_summary(model, report_df=None):
    """Create executive summary report"""
    summary = []
    summary.append("="*60)
    summary.append("CUSTOMER CHURN PREDICTION - EXECUTIVE SUMMARY")
    summary.append("="*60)
    summary.append("")
    
    # Model performance
    summary.append("MODEL PERFORMANCE:")
    metrics = model.metrics['test']
    summary.append(f"  - Model Type: {model.model_type.upper()}")
    summary.append(f"  - Accuracy: {metrics['accuracy']:.2%}")
    summary.append(f"  - Precision: {metrics['precision']:.2%}")
    summary.append(f"  - Recall: {metrics['recall']:.2%}")
    summary.append(f"  - F1 Score: {metrics['f1_score']:.2%}")
    summary.append(f"  - ROC AUC: {metrics['roc_auc']:.2%}")
    summary.append("")
    
    # Confusion matrix interpretation
    summary.append("PREDICTION RESULTS:")
    summary.append(f"  - True Positives: {metrics['true_positives']} (correctly identified churners)")
    summary.append(f"  - False Positives: {metrics['false_positives']} (false alarms)")
    summary.append(f"  - False Negatives: {metrics['false_negatives']} (missed churners)")
    summary.append(f"  - True Negatives: {metrics['true_negatives']} (correctly identified retentions)")
    summary.append("")
    
    # Risk analysis
    if report_df is not None:
        summary.append("RISK ANALYSIS:")
        risk_counts = report_df['risk_category'].value_counts()
        for risk_level in ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk']:
            if risk_level in risk_counts.index:
                count = risk_counts[risk_level]
                pct = count / len(report_df) * 100
                summary.append(f"  - {risk_level}: {count} customers ({pct:.1f}%)")
        summary.append("")
        
        # High risk customers
        high_risk = report_df[report_df['risk_category'].isin(['High Risk', 'Critical Risk'])]
        if len(high_risk) > 0 and 'revenue_monthly' in report_df.columns:
            at_risk_revenue = high_risk['revenue_monthly'].sum()
            summary.append(f"AT-RISK REVENUE:")
            summary.append(f"  - High/Critical risk customers: {len(high_risk)}")
            summary.append(f"  - Monthly revenue at risk: ${at_risk_revenue:,.2f}")
            summary.append("")
    
    # Recommendations
    summary.append("KEY RECOMMENDATIONS:")
    summary.append("  1. Prioritize outreach to Critical and High Risk customers")
    summary.append("  2. Implement retention campaigns for Medium Risk customers")
    summary.append("  3. Focus on improving engagement metrics (login frequency, feature adoption)")
    summary.append("  4. Address payment issues proactively")
    summary.append("  5. Strengthen customer relationships near contract renewal dates")
    summary.append("")
    summary.append("="*60)
    
    summary_text = "\n".join(summary)
    
    # Save to file
    with open(REPORTS_DIR / 'executive_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nExecutive summary saved to {REPORTS_DIR / 'executive_summary.txt'}")
    
    return summary_text


if __name__ == "__main__":
    # Load model and data
    print("Loading model and data...")
    model = joblib.load(MODELS_DIR / 'best_churn_model.pkl')
    fe = joblib.load(MODELS_DIR / 'feature_engineer.pkl')
    
    # Load original data
    df = pd.read_csv(DATA_DIR / 'customer_data.csv')
    
    # For testing, we'll use the test set
    from feature_engineering import prepare_train_test_data
    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_train_test_data(df)
    
    # Get indices for test set to match with original data
    # This is simplified - in production you'd track indices properly
    test_indices = df.index[-len(X_test):]
    test_data = df.iloc[test_indices].reset_index(drop=True)
    
    # Create analyzer
    analyzer = ChurnAnalyzer(model, fe, X_test, y_test, X_train, y_train)
    
    # Generate all visualizations
    analyzer.generate_all_visualizations(test_data)
    
    # Create executive summary
    report_df = pd.read_csv(REPORTS_DIR / 'churn_risk_report.csv')
    create_executive_summary(model, report_df)
