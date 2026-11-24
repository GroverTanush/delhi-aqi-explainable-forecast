"""
dashboard/prediction_service.py
Service layer for loading trained Random Forest models and generating 5-day forecasts.
Reuses existing feature engineering and model assets without retraining.
"""

from datetime import timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent

from src.evaluate import get_aqi_category_6
from src.features import build_feature_row_for_date
from dashboard.data_service import get_clean_daily, get_metrics

# Standard CPCB Health Advisories
ADVISORIES = {
    "Good": "Normal outdoor activities permitted for all groups.",
    "Satisfactory": "Unusually sensitive people should reduce prolonged outdoor exertion.",
    "Moderate": "People with respiratory or cardiac conditions should reduce prolonged outdoor exertion.",
    "Poor": "Avoid prolonged outdoor exertion; sensitive groups should limit time outside.",
    "Very Poor": "Avoid outdoor activity where possible; wear protective filtration masks.",
    "Severe": "Stay indoors and strictly avoid physical exertion; keep windows closed.",
}


@st.cache_resource(show_spinner=False)
def load_rf_models():
    """Loads all 5 trained Random Forest models and feature column definitions."""
    models_dir = REPO_ROOT / "models"
    feat_path = models_dir / "feature_columns.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature columns JSON missing at {feat_path}")
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    models = {}
    for h in range(1, 6):
        m_path = models_dir / f"rf_h{h}.pkl"
        if not m_path.exists():
            raise FileNotFoundError(f"Trained Random Forest model missing at {m_path}")
        models[h] = joblib.load(m_path)

    # Typical test MAE by horizon
    metrics_df = get_metrics()
    rf_metrics = metrics_df[metrics_df["model"] == "Random Forest"].set_index("horizon")
    typical_maes = {h: float(rf_metrics.loc[h, "MAE"]) for h in range(1, 6)}

    # Baseline equivalence note from CI table
    ci_path = REPO_ROOT / "outputs" / "rf_vs_baseline_ci.csv"
    ci_notes = {}
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path).set_index("horizon")
        for h in range(1, 6):
            if h in ci_df.index and bool(ci_df.loc[h, "includes_zero"]):
                ci_notes[h] = "At this horizon the model is statistically indistinguishable from the best baseline."

    return feature_cols, models, typical_maes, ci_notes


def generate_forecast(date_input: str | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Generates a 5-day forward forecast using the actual trained Random Forest models.
    Returns:
        (forecast_df, feature_row, any_alert_exceeded)
    """
    daily_df = get_clean_daily()
    t_dt = pd.to_datetime(date_input)
    date_str_fmt = t_dt.strftime("%Y-%m-%d")

    # Validate that date is present in clean_daily
    if t_dt not in daily_df["date"].values:
        raise ValueError(f"Selected date {date_str_fmt} is not present in historical observations.")

    # Build 38-feature row strictly from historical trailing window
    feat_row = build_feature_row_for_date(daily_df, t_dt)
    feature_cols, models, typical_maes, ci_notes = load_rf_models()

    X_in = feat_row[feature_cols]

    # Map actuals from daily observations
    daily_map = daily_df.set_index("date")["aqi"].to_dict()

    records = []
    any_poor_or_worse = False

    for h in range(1, 6):
        target_dt = t_dt + timedelta(days=h)
        target_str = target_dt.strftime("%Y-%m-%d")

        raw_pred = float(models[h].predict(X_in)[0])
        pred_clipped = float(np.clip(raw_pred, 0.0, 500.0))
        cat = get_aqi_category_6(pred_clipped)
        advisory = ADVISORIES.get(cat, "")
        typical_err = typical_maes[h]

        if pred_clipped > 200.0:
            any_poor_or_worse = True

        actual_aqi = daily_map.get(target_dt, np.nan)
        has_actual = not np.isnan(actual_aqi)
        actual_val_str = f"{actual_aqi:.1f}" if has_actual else "Unavailable"
        abs_err = abs(pred_clipped - actual_aqi) if has_actual else np.nan

        note = ci_notes.get(h, "")

        records.append({
            "horizon": f"Day +{h}",
            "horizon_num": h,
            "forecast_date": date_str_fmt,
            "target_date": target_str,
            "predicted_aqi": round(pred_clipped, 1),
            "category": cat,
            "advisory": advisory,
            "typical_mae": round(typical_err, 1),
            "actual_aqi": actual_val_str,
            "actual_aqi_num": actual_aqi,
            "abs_error": round(abs_err, 1) if not np.isnan(abs_err) else np.nan,
            "has_actual": has_actual,
            "note": note,
        })

    forecast_df = pd.DataFrame(records)
    return forecast_df, feat_row, any_poor_or_worse


@st.cache_data(show_spinner=False)
def explain_single_prediction(date_str: str, horizon: int = 1) -> dict:
    """
    Computes real dynamic TreeExplainer SHAP attribution for a single forecast date and horizon.
    Returns:
        dict with base_value, top_positive_features, top_negative_features, and total_prediction.
    """
    import shap
    daily_df = get_clean_daily()
    t_dt = pd.to_datetime(date_str)
    feat_row = build_feature_row_for_date(daily_df, t_dt)
    feature_cols, models, _, _ = load_rf_models()

    if horizon not in models:
        raise ValueError(f"Horizon {horizon} not supported.")

    rf_model = models[horizon]
    X_in = feat_row[feature_cols]

    expl = shap.TreeExplainer(rf_model)
    shap_vals = expl(X_in)[0]

    base_val = float(shap_vals.base_values)
    values = shap_vals.values
    feat_names = feature_cols
    feat_inputs = X_in.iloc[0].values

    records = []
    for name, s_val, in_val in zip(feat_names, values, feat_inputs):
        records.append({
            "feature": name,
            "shap_value": float(s_val),
            "feature_value": float(in_val),
        })

    df_contrib = pd.DataFrame(records)
    df_pos = df_contrib[df_contrib["shap_value"] > 0].sort_values("shap_value", ascending=False).head(5)
    df_neg = df_contrib[df_contrib["shap_value"] < 0].sort_values("shap_value", ascending=True).head(5)

    return {
        "base_value": round(base_val, 1),
        "prediction": round(float(rf_model.predict(X_in)[0]), 1),
        "top_positive": df_pos.to_dict(orient="records"),
        "top_negative": df_neg.to_dict(orient="records"),
    }
