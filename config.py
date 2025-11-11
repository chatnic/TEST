"""
Configuration file for the Customer Churn Prediction Model
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Data generation parameters
RANDOM_SEED = 42
NUM_CUSTOMERS = 5000

# Feature engineering parameters
CATEGORICAL_FEATURES = [
    'institution_type',
    'region',
    'product_tier',
    'payment_method',
    'support_tier'
]

NUMERICAL_FEATURES = [
    'account_age_months',
    'monthly_transaction_volume',
    'avg_transaction_value',
    'api_calls_per_month',
    'num_active_users',
    'revenue_monthly',
    'support_tickets_6m',
    'days_since_last_login',
    'feature_adoption_score',
    'payment_delay_days',
    'contract_months_remaining',
    'num_integrations'
]

# Model parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CLASS_WEIGHT = 'balanced'

# Model hyperparameters for grid search
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [1, 3, 5]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50, 70],
        'class_weight': ['balanced']
    }
}

# Evaluation thresholds
CHURN_PROBABILITY_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4
