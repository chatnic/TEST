# Quick Start Guide

## 🚀 Get Started in 60 Seconds

### Option 1: Run Everything at Once

```bash
python3 run_full_pipeline.py
```

This will:
1. Generate 5,000 synthetic customer records
2. Train 4 different ML models
3. Select the best performing model
4. Create 6 visualizations
5. Generate predictions for all customers
6. Create executive summary

**Time**: ~30 seconds  
**Output**: All reports in `/workspace/reports/`

---

## 📊 View Results

After running the pipeline, check these files:

### 1. Executive Summary (Start Here!)
```bash
cat reports/executive_summary.txt
```

**Shows**: Model performance, risk analysis, revenue at risk, recommendations

### 2. Customer Risk Report
```bash
head reports/predictions.csv
```

**Shows**: Every customer's churn probability and risk level, sorted by risk

### 3. Visualizations
```bash
ls -lh reports/*.png
```

Open these images to see:
- ROC curve (model accuracy)
- Feature importance (what drives churn)
- Confusion matrix (prediction breakdown)
- Risk distributions
- And more...

---

## 🎯 Common Use Cases

### Scenario 1: Score New Customers

You have a CSV file `new_customers.csv` with customer data:

```bash
python3 predict.py --input new_customers.csv --output scores.csv
```

### Scenario 2: Identify Top 20 At-Risk Customers

```bash
head -21 reports/predictions.csv
```

The file is sorted by churn probability (highest risk first).

### Scenario 3: Calculate Revenue at Risk

```bash
# High + Critical risk customers
python3 -c "
import pandas as pd
df = pd.read_csv('reports/predictions.csv')
at_risk = df[df['risk_level'].isin(['High Risk', 'Critical Risk'])]
print(f'Customers at risk: {len(at_risk)}')
print(f'Revenue at risk: \${at_risk[\"revenue_monthly\"].sum():,.2f}/month')
"
```

### Scenario 4: Export Critical Risk Customers for Outreach

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('reports/predictions.csv')
critical = df[df['risk_level'] == 'Critical Risk']
critical[['customer_id', 'churn_probability', 'revenue_monthly', 'institution_type']].to_csv('critical_customers.csv', index=False)
print(f'Exported {len(critical)} critical risk customers to critical_customers.csv')
"
```

### Scenario 5: Retrain Model with Your Own Data

1. Prepare your CSV with these columns:
   - customer_id, institution_type, region, product_tier
   - account_age_months, monthly_transaction_volume, avg_transaction_value
   - api_calls_per_month, num_active_users, revenue_monthly
   - support_tickets_6m, support_tier, days_since_last_login
   - feature_adoption_score, payment_method, payment_delay_days
   - contract_months_remaining, num_integrations
   - churn (0 or 1 - the target variable)

2. Replace the generated data:
   ```bash
   cp your_data.csv data/customer_data.csv
   ```

3. Retrain:
   ```bash
   python3 model_training.py
   ```

---

## 📈 Understanding the Output

### Risk Levels Explained

| Risk Level | Churn Probability | Action Required |
|------------|-------------------|-----------------|
| **Critical Risk** | 70-100% | Immediate intervention |
| **High Risk** | 50-70% | Proactive outreach within 1 week |
| **Medium Risk** | 30-50% | Monitor and engage |
| **Low Risk** | 0-30% | Standard support |

### Key Metrics Explained

- **ROC AUC** (80.29%): Overall predictive accuracy (80% is good!)
- **Precision** (60.38%): When model predicts churn, it's right 60% of the time
- **Recall** (44.91%): Model catches 45% of actual churners
- **F1 Score** (51.51%): Balance between precision and recall

---

## 🔍 Explore the Data

### View Sample Customer Data
```bash
head -20 data/customer_data.csv
```

### Check Model Performance
```bash
cat reports/model_comparison.csv
```

### See What Matters Most
```bash
cat reports/feature_importance.csv
```

---

## ⚙️ Advanced Usage

### Run Individual Components

```bash
# Step 1: Generate data
python3 data_generator.py

# Step 2: Train models
python3 model_training.py

# Step 3: Create visualizations
python3 visualizations.py

# Step 4: Make predictions
python3 predict.py
```

### Use Python API

```python
from predict import ChurnPredictor

# Load predictor
predictor = ChurnPredictor()

# Predict single customer
customer = {
    'customer_id': 'INST_00001',
    'institution_type': 'Bank',
    'region': 'North America',
    'product_tier': 'Enterprise',
    'account_age_months': 24,
    'monthly_transaction_volume': 50000,
    'avg_transaction_value': 1500,
    'api_calls_per_month': 100000,
    'num_active_users': 500,
    'revenue_monthly': 25000,
    'support_tickets_6m': 5,
    'support_tier': 'Premium',
    'days_since_last_login': 2,
    'feature_adoption_score': 85,
    'payment_method': 'Wire Transfer',
    'payment_delay_days': 0,
    'contract_months_remaining': 18,
    'num_integrations': 12
}

result = predictor.predict_single_customer(customer)
print(f"Churn probability: {result['churn_probability']:.1%}")
print(f"Risk level: {result['risk_level']}")
```

---

## 🆘 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Install dependencies
```bash
pip3 install -r requirements.txt
```

### Issue: "FileNotFoundError" when running predict.py
**Solution**: Run the full pipeline first to generate models
```bash
python3 run_full_pipeline.py
```

### Issue: Predictions seem off
**Solution**: The model is trained on synthetic data. For production use, retrain with your real customer data.

---

## 📚 Learn More

- **Full documentation**: See `README.md`
- **Project overview**: See `PROJECT_SUMMARY.md`
- **Code details**: All Python files are well-commented

---

## 🎯 Next Steps

1. ✅ Run the pipeline: `python3 run_full_pipeline.py`
2. ✅ Review executive summary: `cat reports/executive_summary.txt`
3. ✅ Check visualizations: Open `.png` files in `reports/`
4. ✅ Export high-risk customers for your retention team
5. ✅ Retrain with your real data when ready

**Need help?** All files include detailed comments and documentation.

---

**Built for fintech companies serious about retention** 🚀
