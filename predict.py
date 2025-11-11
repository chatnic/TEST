"""
Make predictions on new customer data
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from config import MODELS_DIR, DATA_DIR, REPORTS_DIR


class ChurnPredictor:
    """Make churn predictions on new data"""
    
    def __init__(self, model_path=None, fe_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (defaults to best model)
            fe_path: Path to saved feature engineer (defaults to standard path)
        """
        if model_path is None:
            model_path = MODELS_DIR / 'best_churn_model.pkl'
        if fe_path is None:
            fe_path = MODELS_DIR / 'feature_engineer.pkl'
        
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        print(f"Loading feature engineer from {fe_path}...")
        self.fe = joblib.load(fe_path)
        
        print("Predictor ready!")
    
    def predict_single_customer(self, customer_data):
        """
        Predict churn for a single customer
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Prepare features
        X = self.fe.prepare_features(df, fit=False)
        
        # Make prediction
        churn_prob = self.model.predict_proba(X)[0]
        churn_pred = self.model.predict(X)[0]
        
        # Determine risk level
        if churn_prob >= 0.7:
            risk_level = "Critical Risk"
        elif churn_prob >= 0.5:
            risk_level = "High Risk"
        elif churn_prob >= 0.3:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
        
        return {
            'customer_id': customer_data.get('customer_id', 'Unknown'),
            'churn_probability': float(churn_prob),
            'will_churn': bool(churn_pred),
            'risk_level': risk_level,
            'model_type': self.model.model_type
        }
    
    def predict_batch(self, data_path, output_path=None):
        """
        Predict churn for multiple customers from CSV
        
        Args:
            data_path: Path to CSV file with customer data
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        print(f"Processing {len(df)} customers...")
        
        # Store customer IDs if present
        customer_ids = df['customer_id'].values if 'customer_id' in df.columns else None
        
        # Prepare features (remove churn if present for prediction)
        df_features = df.drop('churn', axis=1) if 'churn' in df.columns else df
        
        # Process features
        X = self.fe.prepare_features(df_features, fit=False)
        
        # Make predictions
        churn_probs = self.model.predict_proba(X)
        churn_preds = self.model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'churn_probability': churn_probs,
            'predicted_churn': churn_preds
        })
        
        # Add customer IDs if available
        if customer_ids is not None:
            results.insert(0, 'customer_id', customer_ids)
        
        # Add risk categories
        results['risk_level'] = pd.cut(
            results['churn_probability'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
        
        # Add original data for context
        for col in ['institution_type', 'region', 'product_tier', 'revenue_monthly', 
                   'account_age_months', 'days_since_last_login']:
            if col in df.columns:
                results[col] = df[col].values
        
        # Sort by churn probability
        results = results.sort_values('churn_probability', ascending=False)
        
        # Save results
        if output_path is None:
            output_path = REPORTS_DIR / 'predictions.csv'
        
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print prediction summary"""
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total customers: {len(results)}")
        print(f"Predicted to churn: {results['predicted_churn'].sum()} ({results['predicted_churn'].mean()*100:.1f}%)")
        print(f"Average churn probability: {results['churn_probability'].mean():.2%}")
        print("\nRisk Distribution:")
        risk_dist = results['risk_level'].value_counts()
        for risk_level in ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk']:
            if risk_level in risk_dist.index:
                count = risk_dist[risk_level]
                pct = count / len(results) * 100
                print(f"  {risk_level}: {count} customers ({pct:.1f}%)")
        
        if 'revenue_monthly' in results.columns:
            high_risk = results[results['risk_level'].isin(['High Risk', 'Critical Risk'])]
            at_risk_revenue = high_risk['revenue_monthly'].sum()
            print(f"\nMonthly revenue at risk: ${at_risk_revenue:,.2f}")
        
        print("="*60)
        
        # Show top 10 at-risk customers
        print("\nTop 10 Customers at Risk:")
        print("-"*60)
        top_risk = results.head(10)
        for idx, row in top_risk.iterrows():
            cust_id = row.get('customer_id', 'Unknown')
            prob = row['churn_probability']
            risk = row['risk_level']
            print(f"{cust_id}: {prob:.1%} probability - {risk}")
        print("="*60)
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from model"""
        if self.model.feature_importance is None:
            print("Feature importance not available for this model")
            return None
        
        importance_df = self.model.get_feature_importance(self.fe.feature_names, top_n=top_n)
        return importance_df


def main():
    """Command line interface for predictions"""
    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('--input', '-i', type=str, 
                       help='Path to input CSV file (defaults to generated data)')
    parser.add_argument('--output', '-o', type=str,
                       help='Path to save predictions CSV')
    parser.add_argument('--model', '-m', type=str,
                       help='Path to model file (defaults to best model)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ChurnPredictor(model_path=args.model if args.model else None)
    
    # Input file
    input_path = args.input if args.input else DATA_DIR / 'customer_data.csv'
    
    # Output file
    output_path = args.output if args.output else None
    
    # Make predictions
    results = predictor.predict_batch(input_path, output_path)
    
    print("\n✓ Predictions completed successfully!")
    
    return results


if __name__ == "__main__":
    main()
