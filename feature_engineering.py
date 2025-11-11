"""
Feature engineering pipeline for churn prediction model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, 
    TEST_SIZE, VALIDATION_SIZE, RANDOM_SEED,
    MODELS_DIR
)


class FeatureEngineer:
    """Feature engineering and preprocessing for churn prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def create_derived_features(self, df):
        """
        Create additional features from existing data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Revenue per user
        df['revenue_per_user'] = df['revenue_monthly'] / (df['num_active_users'] + 1)
        
        # API calls per user
        df['api_calls_per_user'] = df['api_calls_per_month'] / (df['num_active_users'] + 1)
        
        # Transaction value ratio
        df['transaction_value_ratio'] = (
            df['avg_transaction_value'] / (df['monthly_transaction_volume'] + 1)
        )
        
        # Support tickets per month of account age
        df['support_ticket_rate'] = (
            df['support_tickets_6m'] / (df['account_age_months'] + 1)
        )
        
        # Engagement score (inverse of days since last login)
        df['engagement_score'] = 100 / (df['days_since_last_login'] + 1)
        
        # Integration density
        df['integration_density'] = (
            df['num_integrations'] / (df['account_age_months'] + 1)
        )
        
        # Payment reliability score
        df['payment_reliability'] = np.where(
            df['payment_delay_days'] == 0, 100,
            100 / (df['payment_delay_days'] + 1)
        )
        
        # Contract risk flag
        df['contract_risk'] = (df['contract_months_remaining'] < 3).astype(int)
        
        # High value customer flag
        df['high_value_customer'] = (df['revenue_monthly'] > df['revenue_monthly'].median()).astype(int)
        
        # Low engagement flag
        df['low_engagement'] = (df['days_since_last_login'] > 30).astype(int)
        
        # Payment issues flag
        df['payment_issues'] = (df['payment_delay_days'] > 15).astype(int)
        
        # New customer flag (less than 6 months)
        df['new_customer'] = (df['account_age_months'] < 6).astype(int)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using label encoding
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training data)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unknown categories
                    df[col] = df[col].astype(str)
                    known_labels = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_labels else self.label_encoders[col].classes_[0]
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_numerical_features(self, df, fit=True):
        """
        Scale numerical features using StandardScaler
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scaler (True for training data)
            
        Returns:
            DataFrame with scaled numerical features
        """
        df = df.copy()
        
        # Get all numerical columns (including derived features)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variable if present
        if 'churn' in numerical_cols:
            numerical_cols.remove('churn')
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def prepare_features(self, df, fit=True):
        """
        Complete feature preparation pipeline
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training data)
            
        Returns:
            Processed DataFrame ready for modeling
        """
        # Remove customer ID if present
        if 'customer_id' in df.columns:
            customer_ids = df['customer_id']
            df = df.drop('customer_id', axis=1)
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Store feature names before scaling
        if fit:
            feature_cols = [col for col in df.columns if col != 'churn']
            self.feature_names = feature_cols
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=fit)
        
        return df
    
    def save(self, filepath=None):
        """Save feature engineering pipeline"""
        if filepath is None:
            filepath = MODELS_DIR / 'feature_engineer.pkl'
        joblib.dump(self, filepath)
        print(f"Feature engineer saved to {filepath}")
    
    @staticmethod
    def load(filepath=None):
        """Load feature engineering pipeline"""
        if filepath is None:
            filepath = MODELS_DIR / 'feature_engineer.pkl'
        return joblib.load(filepath)


def prepare_train_test_data(df):
    """
    Split data into train, validation, and test sets
    
    Args:
        df: Input DataFrame with 'churn' column
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_engineer)
    """
    # Separate features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} samples ({y_train.mean()*100:.2f}% churn)")
    print(f"Validation set: {len(X_val)} samples ({y_val.mean()*100:.2f}% churn)")
    print(f"Test set: {len(X_test)} samples ({y_test.mean()*100:.2f}% churn)")
    
    # Initialize and fit feature engineer on training data only
    fe = FeatureEngineer()
    X_train_processed = fe.prepare_features(X_train, fit=True)
    X_val_processed = fe.prepare_features(X_val, fit=False)
    X_test_processed = fe.prepare_features(X_test, fit=False)
    
    # Remove churn column if it exists (shouldn't, but safe check)
    for df_proc in [X_train_processed, X_val_processed, X_test_processed]:
        if 'churn' in df_proc.columns:
            df_proc.drop('churn', axis=1, inplace=True)
    
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, fe


if __name__ == "__main__":
    from config import DATA_DIR
    
    print("Loading customer data...")
    df = pd.read_csv(DATA_DIR / 'customer_data.csv')
    
    print("\nPreparing features...")
    X_train, X_val, X_test, y_train, y_val, y_test, fe = prepare_train_test_data(df)
    
    print(f"\nFeature names: {fe.feature_names}")
    print(f"Total features: {len(fe.feature_names)}")
    
    # Save feature engineer
    fe.save()
