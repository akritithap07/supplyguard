# SupplyGuard Demo Script

### 1. Problem
"Supply-chain companies do not only need to know whether an order will be late. They need to identify risky shipments early, understand why the risk is high, and decide what action to take."

### 2. Dataset
"We use the DataCo SMART Supply Chain dataset. It contains order, customer, product, location and shipping information."

### 3. ML
"We compare Logistic Regression, Decision Tree, Random Forest and XGBoost. We use a time-aware train/test split and evaluate recall, F1, ROC-AUC and PR-AUC instead of relying only on accuracy."

### 4. Leakage
"A key part of our implementation is preventing target leakage. Actual shipping duration and delivery-status information reveal the outcome, so we exclude them from predictors."

### 5. Explainability
"For each shipment, SHAP tells us which model features push the risk higher or lower."

### 6. Decision intelligence
"We convert the probability into Low, Moderate, High and Critical risk bands and provide transparent operational recommendations."

### 7. What-if
"The what-if simulator lets an operator change controllable fields such as shipping mode and compare the model's original and scenario risk."

### 8. Closing
"So SupplyGuard is not just a late-delivery classifier. It is a leakage-aware ML decision-support system that connects prediction, explainability and operational response."
