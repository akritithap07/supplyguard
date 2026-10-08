from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from .analysis import make_eda_plots, write_analysis_summary
from .config import MODEL_DIR, OUTPUT_DIR, PLOTS_DIR, RAW_DATA, TARGET, STRICT_EXCLUDE
from .data_loader import dataset_profile, load_supply_chain_csv, validate_columns
from .feature_engineering import prepare_for_modeling
from .metrics import evaluate_binary
from .preprocessing import build_preprocessor, infer_feature_columns

warnings.filterwarnings("ignore", category=UserWarning)


def chronological_split(df: pd.DataFrame, test_size: float = 0.20):
    """Create a time-aware split when the order timestamp is available."""
    date_col = "order date (DateOrders)"
    if date_col not in df.columns:
        split = int(len(df) * (1 - test_size))
        return df.iloc[:split].copy(), df.iloc[split:].copy()

    dates = pd.to_datetime(df[date_col], errors="coerce")
    ordered = df.assign(_parsed_order_date=dates).sort_values("_parsed_order_date", kind="stable")
    split = int(len(ordered) * (1 - test_size))
    train = ordered.iloc[:split].drop(columns=["_parsed_order_date"])
    test = ordered.iloc[split:].drop(columns=["_parsed_order_date"])
    return train, test


def make_pipeline(df: pd.DataFrame, feature_columns: list[str], model) -> Pipeline:
    preprocessor = build_preprocessor(df, feature_columns)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def save_plots(results: dict, y_test, y_prob, model_dir: Path) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_test, y_prob)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=160)
    plt.close()

    final_metrics = results["metrics"][results["selected_model"]]
    ConfusionMatrixDisplay(confusion_matrix=np.array(final_metrics["confusion_matrix"]), display_labels=[0, 1]).plot()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def build_models(scale_pos_weight: float):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=None),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),
    }


def tune_xgb(base_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    params = {
        "model__max_depth": [4, 5, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__n_estimators": [220, 300, 400],
        "model__subsample": [0.75, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=params,
        n_iter=10,
        scoring="average_precision",
        cv=3,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def main(max_rows: int | None = None, tune: bool = False) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_supply_chain_csv(RAW_DATA)
    validate_columns(df, [TARGET])
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).sort_index()

    # Target and feature engineering.
    df = prepare_for_modeling(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()

    # Remove raw target/outcome fields from predictors by feature inference.
    feature_columns = infer_feature_columns(df)
    train_df, test_df = chronological_split(df)

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET].astype(int)

    positives = max(int(y_train.sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = negatives / positives

    models = build_models(scale_pos_weight)
    metrics = {}
    fitted = {}

    for name, estimator in models.items():
        pipe = make_pipeline(X_train, feature_columns, estimator)
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics[name] = evaluate_binary(y_test, y_pred, y_prob)
        fitted[name] = pipe

    selected_model = max(
        metrics,
        key=lambda name: (metrics[name]["pr_auc"], metrics[name]["recall"], metrics[name]["f1"]),
    )

    # Prefer XGBoost for tuning only when it is already the strongest family or when explicitly requested.
    final_pipeline = fitted[selected_model]
    if tune:
        xgb_pipeline = fitted["XGBoost"]
        tuned = tune_xgb(xgb_pipeline, X_train, y_train)
        tuned_prob = tuned.predict_proba(X_test)[:, 1]
        tuned_pred = (tuned_prob >= 0.5).astype(int)
        tuned_metrics = evaluate_binary(y_test, tuned_pred, tuned_prob)
        metrics["XGBoost (Tuned)"] = tuned_metrics
        fitted["XGBoost (Tuned)"] = tuned
        if (tuned_metrics["pr_auc"], tuned_metrics["recall"], tuned_metrics["f1"]) > (
            metrics[selected_model]["pr_auc"], metrics[selected_model]["recall"], metrics[selected_model]["f1"]
        ):
            selected_model = "XGBoost (Tuned)"
            final_pipeline = tuned

    final_prob = final_pipeline.predict_proba(X_test)[:, 1]
    final_pred = (final_prob >= 0.5).astype(int)

    # Save artifacts.
    joblib.dump(final_pipeline, MODEL_DIR / "model.joblib")
    metadata = {
        "target": TARGET,
        "feature_columns": feature_columns,
        "selected_model": selected_model,
        "risk_thresholds": {"low": 0.30, "moderate": 0.60, "high": 0.80},
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "dataset_profile": dataset_profile(df),
        "excluded_columns_present": sorted([c for c in STRICT_EXCLUDE if c in df.columns]),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    serializable_metrics = {"selected_model": selected_model, "metrics": metrics}
    (MODEL_DIR / "model_metrics.json").write_text(json.dumps(serializable_metrics, indent=2, default=str), encoding="utf-8")

    make_eda_plots(df, PLOTS_DIR)
    write_analysis_summary(df, OUTPUT_DIR / "analysis_summary.json")
    save_plots(serializable_metrics, y_test, final_prob, MODEL_DIR)

    print("\nSupplyGuard training complete")
    print(f"Selected model: {selected_model}")
    print(json.dumps(metrics[selected_model], indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SupplyGuard models.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for faster experimentation.")
    parser.add_argument("--tune", action="store_true", help="Tune XGBoost using randomized search.")
    args = parser.parse_args()
    main(max_rows=args.max_rows, tune=args.tune)
