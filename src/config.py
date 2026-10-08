from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"
DESCRIPTION_DATA = ROOT_DIR / "data" / "raw" / "DescriptionDataCoSupplyChain.csv"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

TARGET = "Late_delivery_risk"
RISK_THRESHOLDS = {
    "Low": (0.00, 0.30),
    "Moderate": (0.30, 0.60),
    "High": (0.60, 0.80),
    "Critical": (0.80, 1.01),
}

# Columns that are not useful for prediction at order/shipment decision time.
STRICT_EXCLUDE = {
    TARGET,
    "Delivery Status",
    "Days for shipping (real)",
    "Shipping date (DateOrders)",
    "Actual Delivery Date",
    "Delivery Date",
    "Order Status",
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Product Image",
}

ID_LIKE_COLUMNS = {
    "Order Id",
    "Order Customer Id",
    "Order Item Id",
    "Order Item Cardprod Id",
    "Customer Id",
    "Product Card Id",
    "Product Category Id",
    "Category Id",
    "Department Id",
}
