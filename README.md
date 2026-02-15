# ML Assignment 2 — Telco Customer Churn (Streamlit)

## a. Problem statement
Build and deploy a Streamlit web application that trains and compares **six classification models** on a public dataset and allows users to upload test CSV data, select a model, and view predictions + evaluation results.

## b. Dataset description
- **Dataset**: Telco Customer Churn
- **Source (Kaggle)**: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target**: `Churn` (Yes/No)
- **Size**: 7043 instances, 21 columns (after dropping `customerID`, **12+ features** remain)
- **Notes**: `TotalCharges` contains blank strings for a few rows; it is converted to numeric during preprocessing.

## c. Models used (comparison table with evaluation metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| Logistic Regression | 0.780 | 0.842 | 0.574 | 0.666 | 0.616 | 0.466 |
| Decision Tree | 0.739 | 0.662 | 0.508 | 0.497 | 0.503 | 0.326 |
| kNN | 0.763 | 0.821 | 0.545 | 0.647 | 0.592 | 0.430 |
| Naive Bayes | 0.718 | 0.808 | 0.481 | 0.791 | 0.598 | 0.429 |
| Random Forest (Ensemble) | 0.763 | 0.823 | 0.545 | 0.652 | 0.594 | 0.432 |
| XGBoost (Ensemble) | 0.777 | 0.841 | 0.563 | 0.714 | 0.630 | 0.480 |

## Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Best AUC (~0.842) and strong overall balance across Precision/Recall; solid baseline for churn. |
| Decision Tree | Lowest AUC and MCC; single tree underperforms ensembles due to higher variance/overfitting risk. |
| kNN | Competitive but slightly below LR/boosting; sensitive to scaling and mixed feature types. |
| Naive Bayes | Highest Recall (~0.791) but lower Precision; tends to predict more churn positives. |
| Random Forest (Ensemble) | Good and stable performance; improves over a single tree with higher MCC and AUC. |
| XGBoost (Ensemble) | Best F1/MCC overall; boosting captures non-linear patterns and feature interactions well. |

## How to run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If the app starts without a saved model artifact, click **Train models now**. Then upload `model/test_data.csv`.


