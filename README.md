# SupplyGuard

## AI-Powered Supply Chain Risk, Delivery Prediction & Decision Intelligence

SupplyGuard is an end-to-end Python machine learning application that predicts late-delivery risk from supply-chain order information and turns the prediction into an operational decision workflow.

Instead of stopping at **Late / On Time**, SupplyGuard provides:

- Probability-based shipment risk scoring
- Low / Moderate / High / Critical risk bands
- Leakage-aware feature engineering
- Model comparison: Logistic Regression, Decision Tree, Random Forest, XGBoost
- SHAP-based explanations for individual predictions
- Regional, market, category and shipping-mode risk analysis
- High-risk shipment explorer
- Transparent rule-based action recommendations
- What-if shipment scenario simulation
- Interactive Streamlit dashboard

> **Important:** This project deliberately avoids using post-outcome variables such as actual shipping duration and delivery status as predictors. These fields can reveal the delivery outcome and cause target leakage.

---

## 1. Dataset

### Primary dataset
**DataCo SMART Supply Chain for Big Data Analysis**

Original source: Mendeley Data, Version 5, DOI `10.17632/8gx2fvg2k6.5`

Download:
- https://data.mendeley.com/datasets/8gx2fvg2k6/5
- Kaggle mirror: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Place:

```text
data/raw/DataCoSupplyChainDataset.csv
```

Recommended additional file:

```text
data/raw/DescriptionDataCoSupplyChain.csv
```

The source describes the structured supply-chain CSV as the main file and the second CSV as the field-description file.

---

## 2. Project architecture

```text
                       DataCo CSV
                           |
                           v
                 +-------------------+
                 | Data Validation   |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Leakage Audit     |
                 +-------------------+
                           |
                           v
                 +-------------------+
                 | Cleaning + FE     |
                 +-------------------+
                           |
              +------------+-------------+
              |            |             |
              v            v             v
          Logistic       Random       XGBoost
          Regression     Forest        (Final)
              |            |             |
              +------------+-------------+
                           |
                           v
                   Model Evaluation
                           |
                           v
                       SHAP XAI
                           |
              +------------+-------------+
              |            |             |
              v            v             v
        Risk Scoring   Recommendations  What-if
              |            |             |
              +------------+-------------+
                           |
                           v
                  Streamlit Dashboard
```

---

## 3. Main ML design

### Target

The project expects a binary late-risk target based on the dataset's `Late_delivery_risk` field, after validating its semantics.

`1 = late / at-risk`

`0 = not late`

The target source is not allowed to appear as an input feature.

### Leakage controls

The following are treated as post-outcome or outcome-revealing and excluded from the feature set:

- `Late_delivery_risk` (target itself)
- `Delivery Status`
- `Days for shipping (real)`
- `Shipping date (DateOrders)`
- actual-delivery timestamps if present
- other explicitly outcome-derived columns detected by the leakage audit

Fields such as customer email, password, names and street are also excluded because they are unnecessary and/or personally identifiable.

### Prediction-time principle

The model should answer:

> **"Given information known at or before the order/shipment decision point, how likely is this shipment to become late?"**

---

## 4. Features

SupplyGuard uses practical order-time features such as:

- Shipping Mode
- Order Priority
- Market
- Order Region
- Order Country
- Category Name
- Department Name
- Customer Segment
- Order Item Quantity
- Order Item Product Price
- Order Item Discount
- Order Item Discount Rate
- Sales
- Order Item Total
- Order Profit Per Order
- Order Item Profit Ratio
- Product Price
- Product Status
- order date components

The feature list is validated against the actual CSV, so missing optional columns do not break the project.

---

## 5. Dashboard

### Executive Overview

KPIs:

- Total orders
- At-risk orders
- Critical orders
- Overall observed delay rate
- High-risk order value exposure
- Highest-risk region

Charts:

- delay trend
- risk by region
- risk by market
- risk by category
- risk by shipping mode

### Shipment Risk Analyzer

Select a shipment and display:

- predicted class
- probability
- risk band
- top risk drivers
- recommended action

### Risk Explorer

Filter and compare:

- Region
- Market
- Product Category
- Shipping Mode
- Customer Segment

### What-if Simulator

Change controllable attributes and compare the original model prediction with a simulated scenario.

### Model Performance

Display:

- model comparison table
- confusion matrix
- ROC curve
- precision-recall curve
- feature importance
- SHAP summary

---

## 6. How to run

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the downloaded CSV in `data/raw/`.

Then:

```powershell
python -m src.train
streamlit run app/streamlit_app.py
```

For faster local experimentation, limit the training rows:

```powershell
python -m src.train --max-rows 80000
```

Tune the final model:

```powershell
python -m src.train --tune
```

---

## 7. Outputs

Training creates:

```text
models/model.joblib
models/metadata.json
models/model_metrics.json
outputs/plots/
outputs/analysis_summary.json
```

---

## 8. Project structure

```text
SupplyGuard/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── DataCoSupplyChainDataset.csv      # user downloads
│   │   └── DescriptionDataCoSupplyChain.csv  # optional
│   └── processed/
│
├── models/
│   ├── model.joblib
│   ├── metadata.json
│   └── model_metrics.json
│
├── notebooks/
│   └── SupplyGuard_End_to_End.ipynb
│
├── outputs/
│   └── plots/
│
├── reports/
│   ├── ARCHITECTURE.md
│   ├── DEMO_SCRIPT.md
│   └── REPORT_OUTLINE.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── metrics.py
│   ├── explainability.py
│   ├── recommendations.py
│   ├── analysis.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_project.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 9. Limitations

- The dataset is historical and synthetic/benchmark-oriented rather than a live company's operational feed.
- Risk scores are model estimates, not guarantees.
- Association-based risk analysis does not prove causal relationships.
- What-if results are model simulations and require operational validation before real deployment.

---

## 10. Future improvements

- Real-time shipment event ingestion
- Supplier-level reliability modeling
- Inventory stockout prediction
- Demand forecasting per SKU
- Route optimization
- Cost-aware intervention optimization
- Model drift monitoring
- API deployment with FastAPI

---

## 11. Dataset citation

Constante, F., Silva, F., & Pereira, A. (2019). *DataCo SMART Supply Chain for Big Data Analysis*, Mendeley Data, Version 5. DOI: 10.17632/8gx2fvg2k6.5.

Follow the license and attribution requirements of the original dataset source.
