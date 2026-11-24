"""
tests/test_dashboard.py
Unit and integration tests for Delhi AQI Intelligence Dashboard services and components.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from dashboard.data_service import (
    get_clean_daily,
    get_metrics,
    get_per_class_metrics,
    get_test_predictions,
    get_dataset_summary,
    get_ci_data,
    get_shap_family_importance,
)
from dashboard.prediction_service import (
    load_rf_models,
    generate_forecast,
    explain_single_prediction,
)
from dashboard.charts import (
    create_forecast_figure,
    create_historical_trend_chart,
    create_particulates_comparison_chart,
    create_monthly_seasonal_chart,
    create_category_distribution_chart,
    create_correlation_heatmap,
    create_evaluation_comparison_bar,
    create_actual_vs_predicted_scatter,
    create_residuals_distribution_chart,
    create_ci_forest_plot,
    create_shap_family_bar,
)


def test_data_service_loads():
    """Verifies that all core data files and summaries load with correct schemas."""
    daily_df = get_clean_daily()
    assert isinstance(daily_df, pd.DataFrame)
    assert len(daily_df) == 3195
    assert "aqi" in daily_df.columns
    assert "date" in daily_df.columns

    metrics_df = get_metrics()
    assert isinstance(metrics_df, pd.DataFrame)
    assert "MAE" in metrics_df.columns
    assert "model" in metrics_df.columns
    assert len(metrics_df) == 25

    summary = get_dataset_summary()
    assert summary["latest_date_str"] == "2026-09-30"
    assert summary["num_stations"] == 35
    assert summary["num_features"] == 38
    assert summary["forecast_horizons"] == 5


def test_models_and_prediction_service():
    """Verifies that RF models load and generate real 5-day forecasts."""
    feat_cols, models, typical_maes, ci_notes = load_rf_models()
    assert len(feat_cols) == 38
    assert len(models) == 5
    for h in range(1, 6):
        assert h in models
        assert h in typical_maes

    # Test prediction on latest base date
    fc_df, feat_row, any_alert = generate_forecast("2026-09-30")
    assert isinstance(fc_df, pd.DataFrame)
    assert len(fc_df) == 5
    assert list(fc_df["horizon"]) == ["Day +1", "Day +2", "Day +3", "Day +4", "Day +5"]
    assert all(0.0 <= val <= 500.0 for val in fc_df["predicted_aqi"])
    assert all(cat in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"] for cat in fc_df["category"])


def test_prediction_with_actuals():
    """Verifies prediction on an earlier date where ground truth exists."""
    test_date = "2025-06-20"
    fc_df, _, _ = generate_forecast(test_date)
    assert len(fc_df) == 5
    assert all(fc_df["has_actual"])
    assert all(isinstance(err, float) and err >= 0 for err in fc_df["abs_error"])


def test_dynamic_shap_attribution():
    """Verifies dynamic TreeExplainer attribution on a single prediction."""
    res = explain_single_prediction("2026-09-30", horizon=1)
    assert "base_value" in res
    assert "prediction" in res
    assert len(res["top_positive"]) > 0 or len(res["top_negative"]) > 0


def test_chart_generation():
    """Verifies that Plotly figures are constructed without exception."""
    daily_df = get_clean_daily()
    summary = get_dataset_summary()
    fc_df, _, _ = generate_forecast(summary["latest_date_str"])

    fig_fc = create_forecast_figure(daily_df, summary["end_dt"], fc_df)
    assert fig_fc is not None

    fig_trend = create_historical_trend_chart(daily_df.head(100), "aqi", "AQI", "Index")
    assert fig_trend is not None

    fig_pm = create_particulates_comparison_chart(daily_df.head(100))
    assert fig_pm is not None

    fig_seas = create_monthly_seasonal_chart(daily_df)
    assert fig_seas is not None

    fig_dist = create_category_distribution_chart(daily_df)
    assert fig_dist is not None

    fig_corr = create_correlation_heatmap(daily_df.head(200))
    assert fig_corr is not None

    metrics_df = get_metrics()
    fig_eval = create_evaluation_comparison_bar(metrics_df, "MAE")
    assert fig_eval is not None

    test_preds = get_test_predictions()
    fig_scat = create_actual_vs_predicted_scatter(test_preds, 1)
    assert fig_scat is not None

    fig_res = create_residuals_distribution_chart(test_preds, 1)
    assert fig_res is not None

    ci_dict = get_ci_data()
    if "baseline" in ci_dict:
        fig_ci = create_ci_forest_plot(ci_dict["baseline"], "Test CI")
        assert fig_ci is not None

    family_df = get_shap_family_importance()
    fig_fam = create_shap_family_bar(family_df, "absolute")
    assert fig_fam is not None


def test_pages_importable():
    """Verifies that all dashboard page modules import cleanly."""
    from dashboard.pages import (
        overview,
        forecast,
        historical,
        explainability,
        evaluation,
        methodology,
    )
    assert overview.render is not None
    assert forecast.render is not None
    assert historical.render is not None
    assert explainability.render is not None
    assert evaluation.render is not None
    assert methodology.render is not None
