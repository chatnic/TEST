# Customer Churn Prediction Model for Fintech

A comprehensive machine learning solution for predicting customer churn among financial institutions using your fintech platform.

## 🎯 Project Overview

This project helps fintech companies predict which financial institution customers are at risk of churning. The model analyzes various behavioral, engagement, and transactional features to identify at-risk customers before they leave, enabling proactive retention efforts.

## 📊 Key Features

- **Synthetic Data Generation**: Creates realistic financial institution customer data for model development
- **Advanced Feature Engineering**: Automatically derives meaningful features from raw customer data
- **Multiple ML Models**: Trains and compares Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Comprehensive Evaluation**: Provides detailed performance metrics and visualizations
- **Risk Categorization**: Classifies customers into Low, Medium, High, and Critical risk categories
- **Batch Predictions**: Easily score new customers and generate actionable reports
- **Executive Reporting**: Generates business-friendly summaries and recommendations

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Generate Training Data
```bash
python data_generator.py
```
This creates a synthetic dataset of 5,000 financial institution customers with realistic churn patterns.

#### 2. Train Models
```bash
python model_training.py
```
This will:
- Train 4 different models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Compare their performance
- Save the best performing model
- Generate performance reports

#### 3. Generate Visualizations and Reports
```bash
python visualizations.py
```
Creates comprehensive visualizations including:
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Confusion matrices
- Risk distribution analysis
- Executive summary report

#### 4. Make Predictions on New Data
```bash
# Predict on default dataset
python predict.py

# Predict on custom dataset
python predict.py --input path/to/customers.csv --output path/to/predictions.csv
```

## 📁 Project Structure

```
/workspace/
├── config.py                  # Configuration and hyperparameters
├── data_generator.py          # Synthetic data generation
├── feature_engineering.py     # Feature preprocessing pipeline
├── model_training.py          # Model training and evaluation
├── visualizations.py          # Analysis and visualization tools
├── predict.py                 # Prediction interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # Data directory
│   └── customer_data.csv      # Generated customer data
│
├── models/                    # Saved models
│   ├── best_churn_model.pkl   # Best performing model
│   └── feature_engineer.pkl   # Feature preprocessing pipeline
│
└── reports/                   # Analysis reports and visualizations
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── feature_importance.png
    ├── churn_risk_report.csv
    ├── executive_summary.txt
    └── model_comparison.csv
```

## 📈 Model Features

The model uses the following customer attributes:

### Demographic Features
- Institution type (Bank, Credit Union, Investment Firm, etc.)
- Geographic region
- Product tier (Basic, Professional, Enterprise)

### Engagement Metrics
- Account age
- Days since last login
- Feature adoption score
- Number of active users
- API usage patterns

### Financial Metrics
- Monthly transaction volume
- Average transaction value
- Monthly revenue
- Payment behavior
- Payment delays

### Support & Integration
- Support tickets (6-month)
- Support tier
- Number of integrations
- Contract months remaining

### Derived Features
The model automatically creates additional features:
- Revenue per user
- API calls per user
- Engagement scores
- Payment reliability metrics
- Risk flags (contract expiration, low engagement, etc.)

## 🎯 Model Performance

The best model typically achieves:
- **ROC AUC**: ~0.85-0.90
- **Precision**: ~0.75-0.85
- **Recall**: ~0.70-0.80
- **F1 Score**: ~0.72-0.82

Performance metrics are saved in `reports/model_comparison.csv` after training.

## 📊 Risk Categories

Customers are classified into four risk levels:

- **Low Risk** (0-30% churn probability): Stable customers
- **Medium Risk** (30-50%): Watch list
- **High Risk** (50-70%): Proactive intervention needed
- **Critical Risk** (70-100%): Immediate action required

## 💡 Business Applications

1. **Proactive Retention**: Identify at-risk customers before they churn
2. **Resource Allocation**: Prioritize retention efforts on high-value at-risk customers
3. **Root Cause Analysis**: Understand key drivers of churn
4. **Campaign Targeting**: Design targeted retention campaigns
5. **Revenue Protection**: Quantify revenue at risk from potential churners

## 🔧 Customization

### Adding New Features

Edit `config.py` to add new features to the model:

```python
NUMERICAL_FEATURES = [
    'your_new_feature',
    # ... existing features
]
```

### Adjusting Model Parameters

Modify hyperparameters in `config.py`:

```python
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        # ... adjust as needed
    }
}
```

### Changing Risk Thresholds

Adjust thresholds in `config.py`:

```python
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4
```

## 📝 Using Real Data

To use your own customer data:

1. Prepare a CSV file with the required features (see `data_generator.py` for schema)
2. Ensure column names match the expected features
3. Run training: `python model_training.py`
4. Make predictions: `python predict.py --input your_data.csv`

## 🔍 Key Insights

Based on the model, the most important churn indicators typically include:

1. **Contract months remaining**: Customers near contract end are high risk
2. **Days since last login**: Engagement is critical
3. **Feature adoption score**: Low adoption indicates dissatisfaction
4. **Payment delays**: Payment issues signal financial or satisfaction problems
5. **Support tickets**: High ticket volume can indicate problems
6. **Account age**: Very new customers have higher churn rates
7. **Number of integrations**: More integrations = stickier customers

## 🚀 Next Steps & Improvements

Potential enhancements:
- Add time-series features (trends over time)
- Implement customer segmentation
- Create a web dashboard for real-time monitoring
- Add survival analysis for time-to-churn predictions
- Build an API for real-time scoring
- Implement automated retraining pipeline
- Add explainability (SHAP values) for individual predictions

## 📞 Support

For questions or issues, review the executive summary in `reports/executive_summary.txt` for insights or examine the detailed reports in the `reports/` directory.

## 📄 License

This project is provided as-is for demonstration and development purposes.

---

**Built for fintech companies serious about customer retention** 🚀
