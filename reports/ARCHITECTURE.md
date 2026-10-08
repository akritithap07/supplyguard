# SupplyGuard Architecture

## High-level system

```text
+---------------------------+
| DataCoSupplyChainDataset  |
+-------------+-------------+
              |
              v
+---------------------------+
| Data Validation           |
| Missing / duplicate check |
+-------------+-------------+
              |
              v
+---------------------------+
| Leakage Audit             |
| Remove outcome-derived    |
| and PII columns           |
+-------------+-------------+
              |
              v
+---------------------------+
| Feature Engineering       |
| Calendar + order metrics  |
+-------------+-------------+
              |
              v
+---------------------------+
| ColumnTransformer         |
| Imputation + OneHot +     |
| Scaling                   |
+-------------+-------------+
              |
              v
+---------------------------+
| ML Models                 |
| LR / DT / RF / XGBoost    |
+-------------+-------------+
              |
              v
+---------------------------+
| Evaluation                |
| F1 / Recall / ROC-AUC     |
| PR-AUC / Confusion Matrix |
+-------------+-------------+
              |
              v
+---------------------------+
| XAI + Decision Layer      |
| SHAP + Risk Bands         |
| Recommendations           |
+-------------+-------------+
              |
              v
+---------------------------+
| Streamlit Application     |
| Overview / Analyzer /     |
| Explorer / What-if        |
+---------------------------+
```
