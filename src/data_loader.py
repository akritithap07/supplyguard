from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_supply_chain_csv(path: str | Path) -> pd.DataFrame:
    """Load DataCo CSV with common encoding fallbacks."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download DataCoSupplyChainDataset.csv "
            "and place it in data/raw/."
        )

    last_error: Exception | None = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"Could not decode {path}: {last_error}")


def validate_columns(df: pd.DataFrame, required: Iterable[str] | None = None) -> None:
    """Raise a useful error when important columns are absent."""
    required = list(required or [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def dataset_profile(df: pd.DataFrame) -> dict:
    """Return lightweight dataset diagnostics for the UI/report."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().sort_values(ascending=False).head(15).to_dict(),
    }
