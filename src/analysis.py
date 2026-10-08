from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def observed_delay_rate(df: pd.DataFrame) -> float | None:
    if "Late_delivery_risk" not in df.columns:
        return None
    return float(pd.to_numeric(df["Late_delivery_risk"], errors="coerce").mean())


def top_risk_groups(df: pd.DataFrame, group_col: str, target_col: str = "Late_delivery_risk", n: int = 10) -> pd.DataFrame:
    if group_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "delay_rate", "orders"])
    x = df.groupby(group_col)[target_col].agg(["mean", "count"]).reset_index()
    x.columns = [group_col, "delay_rate", "orders"]
    return x.sort_values(["delay_rate", "orders"], ascending=[False, False]).head(n)


def make_eda_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Late_delivery_risk" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Late_delivery_risk")
        plt.title("Observed Late Delivery Risk Distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "target_distribution.png", dpi=160)
        plt.close()

    for col, filename, title in [
        ("Shipping Mode", "risk_by_shipping_mode.png", "Observed Risk by Shipping Mode"),
        ("Market", "risk_by_market.png", "Observed Risk by Market"),
        ("Category Name", "risk_by_category.png", "Observed Risk by Product Category"),
        ("Order Region", "risk_by_region.png", "Observed Risk by Region"),
    ]:
        if col in df.columns and "Late_delivery_risk" in df.columns:
            grp = top_risk_groups(df, col, n=10)
            plt.figure(figsize=(9, 5))
            sns.barplot(data=grp, y=col, x="delay_rate", order=grp[col])
            plt.title(title)
            plt.xlabel("Observed late-risk rate")
            plt.ylabel("")
            plt.xlim(0, 1)
            plt.tight_layout()
            plt.savefig(out_dir / filename, dpi=160)
            plt.close()


def write_analysis_summary(df: pd.DataFrame, out_path: Path) -> None:
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "observed_delay_rate": observed_delay_rate(df),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
