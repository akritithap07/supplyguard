from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR
from .explainability import explain_pipeline
from .feature_engineering import prepare_for_modeling
from .recommendations import recommend_actions, risk_band


class SupplyGuardPredictor:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or (MODEL_DIR / "model.joblib")
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Trained model not found. Run `python -m src.train` first."
            )
        self.pipeline = joblib.load(self.model_path)
        metadata_path = MODEL_DIR / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.features = self.metadata.get("feature_columns", [])

    def predict(self, row: pd.DataFrame) -> dict:
        row = prepare_for_modeling(row)
        X = row[self.features].copy()
        probability = float(self.pipeline.predict_proba(X)[:, 1][0])
        prediction = int(probability >= 0.5)
        band = risk_band(probability)
        actions = recommend_actions(probability, row.iloc[0].to_dict())

        try:
            explanations = explain_pipeline(self.pipeline, X)
        except Exception:
            explanations = pd.DataFrame(columns=["feature", "shap_value", "abs_shap"])

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_band": band,
            "actions": actions,
            "explanations": explanations,
        }
