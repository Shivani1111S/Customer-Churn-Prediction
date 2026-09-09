Bank Customer Churn Prediction

Predicting which bank customers are likely to churn, and prioritizing retention outreach by weighting churn probability against account balance rather than using raw churn probability alone.

Problem Statement

Customer attrition is costly for retail banks — acquiring a new customer typically costs far more than retaining an existing one. This project builds a classifier to flag customers at high risk of churning, then goes a step further: instead of just ranking by churn probability, it ranks by potential dollar impact (churn probability × account balance), so retention teams can focus effort where it matters most financially.

Dataset
Source: bank_churn.csv — 10,000 bank customers
Target: Exited (1 = churned, 0 = retained) — imbalanced at ~20% churn / 80% retained
Features: Credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, active membership status, estimated salary
Identifier columns (RowNumber, CustomerId, Surname) were dropped; Gender label-encoded and Geography one-hot encoded
Methodology
Preprocessing — label/one-hot encoding, train/test split (80/20, stratified), feature scaling with StandardScaler
Class imbalance — handled with SMOTE on the training set only (to avoid leakage into the test set)
Models compared:
Logistic Regression (baseline)
XGBoost, tuned via RandomizedSearchCV (25 iterations) over n_estimators, max_depth, learning_rate, subsample, colsample_bytree
Validation — 5-fold stratified cross-validation, scored on ROC-AUC
Threshold optimization — rather than using the default 0.5 cutoff, the classification threshold was chosen to maximize F1 using the precision-recall curve
Explainability — SHAP (TreeExplainer) used on the tuned XGBoost model to identify which features drive individual predictions, not just global feature importance
Business framing — customers ranked by churn_probability × account_balance, surfacing the top-20 customers where losing them would hurt most financially, rather than just the ones most likely to leave
Repository Structure
├── data/
│   └── bank_churn.csv
├── Churn_prediction.ipynb
├── plots/
│   ├── roc_curve.png
│   ├── threshold_optimization.png
│   └── shap_summary.png
├── results/
│   └── (model scores, threshold reports, SHAP rankings, at-risk customer list)
└── README.md

Detailed metrics and scores are available in the results files in this repo (Excel/CSV).

Key Takeaway

Beyond model accuracy, the project's finance-oriented contribution is ranking customers by churn probability weighted by account balance (see the top-20 at-risk customers file in results/) — it reframes churn prediction as a resource-allocation problem: which at-risk customers are worth the retention team's limited time and budget, based on both likelihood of leaving and financial exposure if they do.

Tech Stack

pandas · numpy · scikit-learn · imbalanced-learn (SMOTE) · XGBoost · SHAP · matplotlib

How to Run
bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib
jupyter notebook Churn_prediction.ipynb
