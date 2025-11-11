#!/usr/bin/env python
"""
Complete end-to-end pipeline for customer churn prediction
Runs all steps: data generation, training, evaluation, and reporting
"""

import sys
from pathlib import Path

def main():
    """Execute full pipeline"""
    
    print("\n" + "="*70)
    print(" CUSTOMER CHURN PREDICTION - FULL PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Step 1: Generate Data
        print("STEP 1: Generating synthetic customer data...")
        print("-"*70)
        from data_generator import generate_customer_data, save_data
        df = generate_customer_data()
        save_data(df)
        print("✓ Data generation completed\n")
        
        # Step 2: Train Models
        print("\nSTEP 2: Training and evaluating models...")
        print("-"*70)
        from model_training import main as train_main
        models, best_model, fe = train_main()
        print("✓ Model training completed\n")
        
        # Step 3: Generate Visualizations
        print("\nSTEP 3: Generating visualizations and reports...")
        print("-"*70)
        from config import DATA_DIR, MODELS_DIR
        import pandas as pd
        import joblib
        from feature_engineering import prepare_train_test_data
        from visualizations import ChurnAnalyzer, create_executive_summary
        
        # Load data
        df = pd.read_csv(DATA_DIR / 'customer_data.csv')
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_train_test_data(df)
        
        # Get test data indices (simplified - in production track properly)
        test_indices = df.index[-len(X_test):]
        test_data = df.iloc[test_indices].reset_index(drop=True)
        
        # Create analyzer
        analyzer = ChurnAnalyzer(best_model, fe, X_test, y_test, X_train, y_train)
        
        # Generate all visualizations
        analyzer.generate_all_visualizations(test_data)
        
        # Create executive summary
        from config import REPORTS_DIR
        report_df = pd.read_csv(REPORTS_DIR / 'churn_risk_report.csv')
        create_executive_summary(best_model, report_df)
        
        print("✓ Visualization generation completed\n")
        
        # Step 4: Generate Predictions
        print("\nSTEP 4: Generating predictions for all customers...")
        print("-"*70)
        from predict import ChurnPredictor
        
        predictor = ChurnPredictor()
        results = predictor.predict_batch(DATA_DIR / 'customer_data.csv')
        
        print("✓ Predictions completed\n")
        
        # Final Summary
        print("\n" + "="*70)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Generated Files:")
        print(f"   • Data: {DATA_DIR / 'customer_data.csv'}")
        print(f"   • Best Model: {MODELS_DIR / 'best_churn_model.pkl'}")
        print(f"   • Reports: {REPORTS_DIR}/")
        print(f"     - executive_summary.txt")
        print(f"     - churn_risk_report.csv")
        print(f"     - model_comparison.csv")
        print(f"     - feature_importance.csv")
        print(f"     - predictions.csv")
        print(f"   • Visualizations: {REPORTS_DIR}/")
        print(f"     - confusion_matrix.png")
        print(f"     - roc_curve.png")
        print(f"     - precision_recall_curve.png")
        print(f"     - feature_importance.png")
        print(f"     - probability_distribution.png")
        print(f"     - threshold_analysis.png")
        
        print("\n📊 Next Steps:")
        print("   1. Review the executive summary: reports/executive_summary.txt")
        print("   2. Examine visualizations in the reports/ directory")
        print("   3. Check high-risk customers in churn_risk_report.csv")
        print("   4. Use predict.py to score new customers")
        
        print("\n🎯 Model Performance:")
        metrics = best_model.metrics['test']
        print(f"   • ROC AUC: {metrics['roc_auc']:.2%}")
        print(f"   • F1 Score: {metrics['f1_score']:.2%}")
        print(f"   • Precision: {metrics['precision']:.2%}")
        print(f"   • Recall: {metrics['recall']:.2%}")
        
        print("\n" + "="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
