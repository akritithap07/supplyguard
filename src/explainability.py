from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def transformed_feature_names(preprocessor) -> np.ndarray:
    """Return names after OneHotEncoding/ColumnTransformer transformation."""
    return np.asarray(preprocessor.get_feature_names_out())


def explain_pipeline(pipeline, X_row: pd.DataFrame) -> pd.DataFrame:
    """Explain one prediction. Uses SHAP for tree models and a coefficient-based fallback for linear models."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    names = transformed_feature_names(preprocessor)
    model_name = type(model).__name__.lower()

    if hasattr(model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_transformed)
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, 0]
            values = values[0]
        except Exception:
            values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_).reshape(-1)
        values = values * X_transformed[0]
    else:
        return pd.DataFrame(columns=["feature", "shap_value", "abs_shap"])

    if len(values) != len(names):
        return pd.DataFrame(columns=["feature", "shap_value", "abs_shap"])

    result = pd.DataFrame({
        "feature": names,
        "shap_value": values,
        "abs_shap": np.abs(values),
    }).sort_values("abs_shap", ascending=False)

    return result.head(15).reset_index(drop=True)
