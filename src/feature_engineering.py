from __future__ import annotations

import pandas as pd


def add_order_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features from the order timestamp without using future information."""
    df = df.copy()
    date_col = "order date (DateOrders)"
    if date_col not in df.columns:
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["order_year"] = dt.dt.year
    df["order_month"] = dt.dt.month
    df["order_quarter"] = dt.dt.quarter
    df["order_day_of_week"] = dt.dt.dayofweek
    df["order_day"] = dt.dt.day
    df["order_weekofyear"] = dt.dt.isocalendar().week.astype("float")
    return df


def add_safe_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic order-level features from fields available at order time."""
    df = df.copy()

    if {"Order Item Product Price", "Order Item Quantity"}.issubset(df.columns):
        df["order_value_estimate"] = (
            pd.to_numeric(df["Order Item Product Price"], errors="coerce")
            * pd.to_numeric(df["Order Item Quantity"], errors="coerce")
        )

    if {"Order Item Discount", "Order Item Product Price"}.issubset(df.columns):
        price = pd.to_numeric(df["Order Item Product Price"], errors="coerce")
        discount = pd.to_numeric(df["Order Item Discount"], errors="coerce")
        df["discount_to_price"] = (discount / price.replace(0, pd.NA)).astype("float64")

    return df


def prepare_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature engineering before model column selection."""
    return add_safe_numeric_features(add_order_date_features(df))
