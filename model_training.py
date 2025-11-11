"""
Model training and evaluation for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from datetime import datetime
from config import MODELS_DIR, REPORTS_DIR, DATA_DIR, RANDOM_SEED
from feature_engineering import prepare_train_test_data


class ChurnModel:
    """Churn prediction model wrapper"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize churn model
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm', 'logistic')
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.feature_importance = None
        self.metrics = {}
        
    def _initialize_model(self):
        """Initialize the selected model with optimal parameters"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=50,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                class_weight='balanced',
                random_state=RANDOM_SEED,
                max_iter=1000,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for XGBoost/LightGBM early stopping)
            y_val: Validation labels (optional)
        """
        print(f"\nTraining {self.model_type} model...")
        
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            # Use early stopping for gradient boosting models
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])
        
        print(f"{self.model_type} model training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X, y, dataset_name='test'):
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating on {dataset_name} set...")
        
        # Predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'avg_precision': average_precision_score(y, y_pred_proba),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Store metrics
        self.metrics[dataset_name] = metrics
        
        # Print results
        print(f"\n{dataset_name.upper()} SET RESULTS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['avg_precision']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}")
        print(f"FN: {metrics['false_negatives']}, TP: {metrics['true_positives']}")
        
        return metrics
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get top N most important features
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, filepath=None):
        """Save model to disk"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = MODELS_DIR / f'churn_model_{self.model_type}_{timestamp}.pkl'
        
        joblib.dump(self, filepath)
        print(f"\nModel saved to {filepath}")
        return filepath
    
    @staticmethod
    def load(filepath):
        """Load model from disk"""
        return joblib.load(filepath)


def train_multiple_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """
    Train and compare multiple model types
    
    Returns:
        Dictionary of trained models with their metrics
    """
    models = {}
    model_types = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        try:
            # Initialize and train model
            model = ChurnModel(model_type=model_type)
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate on all datasets
            model.evaluate(X_train, y_train, 'train')
            model.evaluate(X_val, y_val, 'validation')
            model.evaluate(X_test, y_test, 'test')
            
            # Get feature importance
            if model.feature_importance is not None:
                importance_df = model.get_feature_importance(feature_names, top_n=20)
                print(f"\nTop 20 Most Important Features for {model_type}:")
                print(importance_df.to_string(index=False))
            
            # Save model
            model.save()
            
            models[model_type] = model
            
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            continue
    
    # Compare models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON (Test Set)")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame([
        {
            'model': model_type,
            'accuracy': model.metrics['test']['accuracy'],
            'precision': model.metrics['test']['precision'],
            'recall': model.metrics['test']['recall'],
            'f1_score': model.metrics['test']['f1_score'],
            'roc_auc': model.metrics['test']['roc_auc'],
        }
        for model_type, model in models.items()
    ]).sort_values('roc_auc', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)
    
    # Select best model based on ROC AUC
    best_model_type = comparison_df.iloc[0]['model']
    best_model = models[best_model_type]
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_type.upper()}")
    print(f"Test ROC AUC: {best_model.metrics['test']['roc_auc']:.4f}")
    print(f"{'='*60}")
    
    # Save best model with special name
    best_model_path = MODELS_DIR / 'best_churn_model.pkl'
    joblib.dump(best_model, best_model_path)
    print(f"\nBest model saved to {best_model_path}")
    
    return models, best_model


def main():
    """Main training pipeline"""
    print("="*60)
    print("CUSTOMER CHURN PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_DIR / 'customer_data.csv')
    print(f"Loaded {len(df)} customers")
    print(f"Churn rate: {df['churn'].mean()*100:.2f}%")
    
    # Prepare features
    print("\nPreparing features...")
    X_train, X_val, X_test, y_train, y_val, y_test, fe = prepare_train_test_data(df)
    
    # Save feature engineer
    fe.save()
    
    print(f"\nTotal features: {len(fe.feature_names)}")
    
    # Train models
    models, best_model = train_multiple_models(
        X_train, y_train, X_val, y_val, X_test, y_test, fe.feature_names
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}")
    
    return models, best_model, fe


if __name__ == "__main__":
    models, best_model, fe = main()
