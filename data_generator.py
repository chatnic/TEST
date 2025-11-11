"""
Synthetic data generator for financial institution customers
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import RANDOM_SEED, NUM_CUSTOMERS, DATA_DIR

np.random.seed(RANDOM_SEED)


def generate_customer_data(num_customers=NUM_CUSTOMERS):
    """
    Generate synthetic customer data for financial institutions
    
    Args:
        num_customers: Number of customer records to generate
        
    Returns:
        DataFrame with customer features and churn labels
    """
    
    # Customer IDs
    customer_ids = [f"INST_{i:05d}" for i in range(num_customers)]
    
    # Institution types
    institution_types = np.random.choice(
        ['Bank', 'Credit Union', 'Investment Firm', 'Insurance Company', 'Payment Processor'],
        size=num_customers,
        p=[0.35, 0.20, 0.15, 0.15, 0.15]
    )
    
    # Geographic regions
    regions = np.random.choice(
        ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East'],
        size=num_customers,
        p=[0.40, 0.25, 0.20, 0.10, 0.05]
    )
    
    # Product tiers
    product_tiers = np.random.choice(
        ['Basic', 'Professional', 'Enterprise'],
        size=num_customers,
        p=[0.30, 0.50, 0.20]
    )
    
    # Account age (in months)
    account_age_months = np.random.exponential(scale=24, size=num_customers)
    account_age_months = np.clip(account_age_months, 1, 120).astype(int)
    
    # Transaction metrics
    monthly_transaction_volume = np.random.lognormal(mean=8, sigma=1.5, size=num_customers)
    monthly_transaction_volume = np.clip(monthly_transaction_volume, 10, 1000000).astype(int)
    
    avg_transaction_value = np.random.lognormal(mean=6, sigma=1.2, size=num_customers)
    avg_transaction_value = np.clip(avg_transaction_value, 50, 50000).round(2)
    
    # API usage
    api_calls_per_month = np.random.lognormal(mean=10, sigma=1.5, size=num_customers)
    api_calls_per_month = np.clip(api_calls_per_month, 100, 10000000).astype(int)
    
    # Active users per institution
    num_active_users = np.random.lognormal(mean=5, sigma=1.8, size=num_customers)
    num_active_users = np.clip(num_active_users, 1, 10000).astype(int)
    
    # Revenue (monthly)
    revenue_monthly = np.random.lognormal(mean=8.5, sigma=1.5, size=num_customers)
    revenue_monthly = np.clip(revenue_monthly, 500, 500000).round(2)
    
    # Support metrics
    support_tickets_6m = np.random.poisson(lam=8, size=num_customers)
    support_tickets_6m = np.clip(support_tickets_6m, 0, 100)
    
    support_tier = np.random.choice(
        ['Standard', 'Priority', 'Premium'],
        size=num_customers,
        p=[0.50, 0.35, 0.15]
    )
    
    # Engagement metrics
    days_since_last_login = np.random.exponential(scale=7, size=num_customers)
    days_since_last_login = np.clip(days_since_last_login, 0, 90).astype(int)
    
    feature_adoption_score = np.random.beta(a=5, b=2, size=num_customers) * 100
    feature_adoption_score = np.clip(feature_adoption_score, 0, 100).round(1)
    
    # Payment behavior
    payment_method = np.random.choice(
        ['Credit Card', 'Wire Transfer', 'ACH', 'Invoice'],
        size=num_customers,
        p=[0.25, 0.30, 0.35, 0.10]
    )
    
    payment_delay_days = np.random.exponential(scale=5, size=num_customers)
    payment_delay_days = np.clip(payment_delay_days, 0, 60).astype(int)
    
    # Contract information
    contract_months_remaining = np.random.uniform(0, 36, size=num_customers).astype(int)
    
    # Integration count
    num_integrations = np.random.poisson(lam=5, size=num_customers)
    num_integrations = np.clip(num_integrations, 0, 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'institution_type': institution_types,
        'region': regions,
        'product_tier': product_tiers,
        'account_age_months': account_age_months,
        'monthly_transaction_volume': monthly_transaction_volume,
        'avg_transaction_value': avg_transaction_value,
        'api_calls_per_month': api_calls_per_month,
        'num_active_users': num_active_users,
        'revenue_monthly': revenue_monthly,
        'support_tickets_6m': support_tickets_6m,
        'support_tier': support_tier,
        'days_since_last_login': days_since_last_login,
        'feature_adoption_score': feature_adoption_score,
        'payment_method': payment_method,
        'payment_delay_days': payment_delay_days,
        'contract_months_remaining': contract_months_remaining,
        'num_integrations': num_integrations
    })
    
    # Generate churn labels based on realistic business rules
    churn_probability = calculate_churn_probability(df)
    df['churn'] = (np.random.random(num_customers) < churn_probability).astype(int)
    
    return df


def calculate_churn_probability(df):
    """
    Calculate churn probability based on customer features
    Uses realistic business logic for financial institutions
    """
    
    # Base probability
    prob = np.full(len(df), 0.15)
    
    # Account age - newer customers more likely to churn
    prob += np.where(df['account_age_months'] < 6, 0.25, 0)
    prob += np.where(df['account_age_months'] < 12, 0.15, 0)
    prob -= np.where(df['account_age_months'] > 24, 0.10, 0)
    
    # Low engagement increases churn
    prob += np.where(df['days_since_last_login'] > 30, 0.30, 0)
    prob += np.where(df['days_since_last_login'] > 60, 0.20, 0)
    
    # Low feature adoption
    prob += np.where(df['feature_adoption_score'] < 30, 0.25, 0)
    prob -= np.where(df['feature_adoption_score'] > 70, 0.15, 0)
    
    # Payment issues
    prob += np.where(df['payment_delay_days'] > 15, 0.20, 0)
    prob += np.where(df['payment_delay_days'] > 30, 0.25, 0)
    
    # High support tickets indicate dissatisfaction
    prob += np.where(df['support_tickets_6m'] > 15, 0.20, 0)
    
    # Low transaction volume
    prob += np.where(df['monthly_transaction_volume'] < 1000, 0.20, 0)
    
    # Contract expiration
    prob += np.where(df['contract_months_remaining'] < 3, 0.30, 0)
    prob += np.where(df['contract_months_remaining'] == 0, 0.40, 0)
    
    # Low integration count (less sticky)
    prob += np.where(df['num_integrations'] < 2, 0.15, 0)
    prob -= np.where(df['num_integrations'] > 8, 0.15, 0)
    
    # Product tier
    prob += np.where(df['product_tier'] == 'Basic', 0.10, 0)
    prob -= np.where(df['product_tier'] == 'Enterprise', 0.15, 0)
    
    # Low revenue customers
    prob += np.where(df['revenue_monthly'] < 2000, 0.15, 0)
    
    # Clip to valid probability range
    prob = np.clip(prob, 0.01, 0.95)
    
    return prob


def save_data(df, filename='customer_data.csv'):
    """Save generated data to CSV file"""
    filepath = DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    print(f"Total customers: {len(df)}")
    print(f"Churned customers: {df['churn'].sum()} ({df['churn'].mean()*100:.2f}%)")
    print(f"Active customers: {(1-df['churn']).sum()} ({(1-df['churn'].mean())*100:.2f}%)")
    return filepath


if __name__ == "__main__":
    print("Generating synthetic customer data for financial institutions...")
    df = generate_customer_data()
    save_data(df)
    
    # Display sample
    print("\nSample data:")
    print(df.head(10))
    print("\nData statistics:")
    print(df.describe())
