"""
dashboard/data_service.py
Data access layer with caching for clean datasets, model outputs, and metrics.
Uses relative paths anchored to the project root.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner=False)
def get_clean_daily() -> pd.DataFrame:
    """Loads clean_daily.csv with datetime parsed and index ready."""
    path = REPO_ROOT / "data" / "processed" / "clean_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data file: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def get_metrics() -> pd.DataFrame:
    """Loads regression and classification metrics from outputs/metrics.csv."""
    path = REPO_ROOT / "outputs" / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation metrics file: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_per_class_metrics() -> pd.DataFrame:
    """Loads per_class_metrics.csv or computes it from test_predictions.csv if missing."""
    path = REPO_ROOT / "outputs" / "per_class_metrics.csv"
    if path.exists():
        return pd.read_csv(path)

    # Compute dynamically if missing
    preds_path = REPO_ROOT / "outputs" / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Neither per_class_metrics.csv nor test_predictions.csv found at {path}")

    from src.evaluate import per_class_report
    df = pd.read_csv(preds_path)
    models_map = {
        "Persistence": "pred_persistence",
        "Climatology": "pred_climatology",
        "Damped Anomaly Persistence": "pred_damped_persistence",
        "Ridge": "pred_ridge",
        "Random Forest": "pred_random_forest",
    }
    records = []
    for h in range(1, 6):
        sub = df[df["horizon"] == h]
        for model_name, col in models_map.items():
            rep = per_class_report(sub["truth"], sub[col])
            rep["model"] = model_name
            rep["horizon"] = h
            records.append(rep)
    res = pd.concat(records, ignore_index=True)
    res = res[["model", "horizon", "category", "precision", "recall", "f1", "support"]]
    res.to_csv(path, index=False)
    return res


@st.cache_data(show_spinner=False)
def get_test_predictions() -> pd.DataFrame:
    """Loads outputs/test_predictions.csv with date parsed."""
    path = REPO_ROOT / "outputs" / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing test predictions file: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def get_ci_data() -> dict[str, pd.DataFrame]:
    """Loads paired moving-block bootstrap confidence intervals."""
    out_dir = REPO_ROOT / "outputs"
    res = {}
    
    baseline_ci = out_dir / "rf_vs_baseline_ci.csv"
    if baseline_ci.exists():
        res["baseline"] = pd.read_csv(baseline_ci)

    ridge_ci = out_dir / "rf_vs_ridge_ci.csv"
    if ridge_ci.exists():
        res["ridge"] = pd.read_csv(ridge_ci)

    winter_ci = out_dir / "winter_split_ci.csv"
    if winter_ci.exists():
        res["winter"] = pd.read_csv(winter_ci)

    return res


@st.cache_data(show_spinner=False)
def get_shap_family_importance() -> pd.DataFrame:
    """Loads outputs/shap_family_importance.csv."""
    path = REPO_ROOT / "outputs" / "shap_family_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing SHAP family importance file: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_worst_days() -> dict[int, pd.DataFrame]:
    """Loads outputs/worst_days_h*.csv."""
    out_dir = REPO_ROOT / "outputs"
    res = {}
    for h in range(1, 6):
        p = out_dir / f"worst_days_h{h}.csv"
        if p.exists():
            res[h] = pd.read_csv(p)
    return res


@st.cache_data(show_spinner=False)
def get_dataset_summary() -> dict:
    """Returns verified dataset metadata and latest observation values."""
    daily_df = get_clean_daily()
    latest_row = daily_df.sort_values("date").iloc[-1]
    
    return {
        "start_date": daily_df["date"].min().strftime("%Y-%m-%d"),
        "end_date": daily_df["date"].max().strftime("%Y-%m-%d"),
        "start_dt": daily_df["date"].min(),
        "end_dt": daily_df["date"].max(),
        "total_days": len(daily_df),
        "valid_aqi_days": int(daily_df["aqi"].notna().sum()),
        "num_stations": 35,
        "num_features": 38,
        "forecast_horizons": 5,
        "evaluation_samples": 887,
        "train_rows": 1887,
        "val_rows": 352,
        "latest_date_str": latest_row["date"].strftime("%Y-%m-%d"),
        "latest_aqi": float(latest_row["aqi"]),
        "latest_category": str(latest_row["aqi_category"]),
        "latest_pm25": float(latest_row["pm25"]),
        "latest_pm10": float(latest_row["pm10"]),
        "latest_row": latest_row,
    }
