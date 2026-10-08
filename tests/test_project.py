import pandas as pd

from src.feature_engineering import prepare_for_modeling
from src.recommendations import risk_band, recommend_actions


def test_feature_engineering_adds_order_value():
    df = pd.DataFrame({
        "Order Item Product Price": [100.0],
        "Order Item Quantity": [3],
    })
    out = prepare_for_modeling(df)
    assert out.loc[0, "order_value_estimate"] == 300.0


def test_risk_bands():
    assert risk_band(0.10) == "Low"
    assert risk_band(0.45) == "Moderate"
    assert risk_band(0.70) == "High"
    assert risk_band(0.95) == "Critical"


def test_recommendation_returns_action():
    actions = recommend_actions(0.90, {"Shipping Mode": "Standard Class"})
    assert actions
