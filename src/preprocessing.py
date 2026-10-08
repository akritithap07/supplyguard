from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_LIKE_COLUMNS, STRICT_EXCLUDE

# Deliberately conservative feature set: high-cardinality identifiers, PII and outcome-derived
# fields are not used. Optional fields are selected only if they exist in the real file.
PREFERRED_FEATURES = [
    "Shipping Mode",
    "Order Priority",
    "Market",
    "Order Region",
    "Order Country",
    "Customer Segment",
    "Category Name",
    "Department Name",
    "Product Name",
    "Product Status",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Total",
    "Order Profit Per Order",
    "Sales",
    "order_value_estimate",
    "discount_to_price",
    "order_year",
    "order_month",
    "order_quarter",
    "order_day_of_week",
    "order_day",
    "order_weekofyear",
    "Latitude",
    "Longitude",
]


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select a controlled, leakage-aware feature set from whatever columns exist."""
    columns = []
    for col in PREFERRED_FEATURES:
        if col in df.columns and col not in STRICT_EXCLUDE and col not in ID_LIKE_COLUMNS:
            columns.append(col)
    if not columns:
        raise ValueError("No usable model features were found. Check the dataset/schema.")
    return columns


def split_feature_types(df: pd.DataFrame, feature_columns: Iterable[str]) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame, feature_columns: Iterable[str]) -> ColumnTransformer:
    numeric, categorical = split_feature_types(df, feature_columns)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
