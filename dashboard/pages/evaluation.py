"""
dashboard/pages/evaluation.py
Page 5: Model Evaluation & Benchmarking
Rigorous, scientifically honest comparison of Random Forest against Ridge and 3 baseline models
(Persistence, Climatology, Damped Anomaly Persistence) across 887 test evaluation days.
Includes paired moving-block bootstrap confidence intervals, classification metrics, and residual error analysis.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.components import render_header, render_freshness_banner
from dashboard.data_service import (
    get_metrics,
    get_per_class_metrics,
    get_test_predictions,
    get_ci_data,
    get_worst_days,
    get_dataset_summary,
)
from dashboard.charts import (
    create_evaluation_comparison_bar,
    create_actual_vs_predicted_scatter,
    create_residuals_distribution_chart,
    create_ci_forest_plot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def render():
    summary = get_dataset_summary()
    metrics_df = get_metrics()
    per_class_df = get_per_class_metrics()
    test_preds = get_test_predictions()
    ci_dict = get_ci_data()
    worst_dict = get_worst_days()

    render_header(
        title="Model Evaluation & Statistical Rigor",
        subtitle="Empirical comparison of Random Forest against Ridge and classical time-series baselines on the 887-day test set.",
        badge="Scientific Benchmarking",
    )

    render_freshness_banner(summary["latest_date_str"])

    # Scientific Honesty Notice
    st.markdown(
        """
        <div style="background-color: #FAFBF9; border: 1px solid #E3E8E2; border-left: 4px solid #246B57; border-radius: 6px; padding: 0.9rem 1.15rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 600; font-size: 0.92rem; color: #202923; margin-bottom: 0.25rem;">
                Scientific Honesty & Empirical Reality
            </div>
            <div style="font-size: 0.84rem; color: #4A554E; line-height: 1.45;">
                In accordance with rigorous statistical practices, we report actual model performance without selective cherry-picking. 
                While <strong>Random Forest is clearly superior at Horizon Day +1</strong> (MAE 27.57 vs 28.90 best baseline, bootstrap p &lt; 0.05), 
                at horizons Day +2 through Day +4 its 95% confidence interval crosses zero, indicating <strong>statistical equivalence</strong> 
                to Damped Anomaly Persistence. At Day +5, simple seasonal Climatology achieves a slightly lower MAE (40.49 vs 41.48).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section A: Regression Metrics Table ────────────────────────────────────
    st.markdown("### A. Multi-Horizon Regression Metrics")
    st.caption("Evaluated strictly on the held-out test partition (2024-04-06 to 2026-09-25; N=887 days).")

    reg_cols = ["model", "horizon", "MAE", "RMSE", "R2", "bias", "accuracy", "within_one_category_accuracy", "alert_precision", "alert_recall"]
    reg_df = metrics_df[reg_cols].copy()
    reg_df.columns = [
        "Model", "Horizon", "MAE (AQI)", "RMSE (AQI)", "R²", "Mean Bias",
        "Exact Cat Acc", "±1 Cat Acc", "Alert Prec", "Alert Rec"
    ]
    reg_df["Horizon"] = reg_df["Horizon"].apply(lambda h: f"Day +{h}")
    reg_df["MAE (AQI)"] = reg_df["MAE (AQI)"].round(2)
    reg_df["RMSE (AQI)"] = reg_df["RMSE (AQI)"].round(2)
    reg_df["R²"] = reg_df["R²"].round(3)
    reg_df["Mean Bias"] = reg_df["Mean Bias"].round(2)
    reg_df["Exact Cat Acc"] = (reg_df["Exact Cat Acc"] * 100).round(1).apply(lambda v: f"{v}%")
    reg_df["±1 Cat Acc"] = (reg_df["±1 Cat Acc"] * 100).round(1).apply(lambda v: f"{v}%")
    reg_df["Alert Prec"] = (reg_df["Alert Prec"] * 100).round(1).apply(lambda v: f"{v}%")
    reg_df["Alert Rec"] = (reg_df["Alert Rec"] * 100).round(1).apply(lambda v: f"{v}%")

    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    # ── Section B: Visual Benchmarking ─────────────────────────────────────────
    st.markdown("### B. Visual Model Benchmarking")
    c_m1, c_m2 = st.columns(2)

    with c_m1:
        fig_mae = create_evaluation_comparison_bar(metrics_df, metric="MAE")
        st.plotly_chart(fig_mae, use_container_width=True)

    with c_m2:
        fig_r2 = create_evaluation_comparison_bar(metrics_df, metric="R2")
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")

    # ── Section C: Paired Block-Bootstrap Confidence Intervals ─────────────────
    st.markdown("### C. Statistical Significance: Moving-Block Bootstrap CIs")
    st.caption(
        "Autocorrelation in daily air quality invalidates standard i.i.d. t-tests. We apply a 7-day moving-block bootstrap (B=2,000 resamples) "
        "on paired daily error differences Δ = |e_RF| - |e_Base|. If the 95% interval is entirely negative, RF is clearly superior. If it crosses 0, the models are statistically equivalent."
    )

    if "baseline" in ci_dict:
        b_ci = ci_dict["baseline"]
        c_forest, c_ci_tbl = st.columns([1.2, 1])

        with c_forest:
            fig_ci = create_ci_forest_plot(b_ci, title="Random Forest vs. Best Baseline (95% CI)")
            st.plotly_chart(fig_ci, use_container_width=True)

        with c_ci_tbl:
            st.markdown("##### Bootstrap CI Summary (RF vs. Best Baseline)")
            ci_disp = b_ci.copy()
            ci_disp["Horizon"] = ci_disp["horizon"].apply(lambda h: f"Day +{h}")
            ci_disp["Δ MAE"] = ci_disp["mean_diff_mae"].round(2)
            ci_disp["95% CI"] = ci_disp.apply(lambda r: f"[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]", axis=1)
            ci_disp["Conclusion"] = ci_disp.apply(
                lambda r: "RF Clearly Better (p < 0.05)" if r["rf_clearly_better"] else "Statistically Indistinguishable" if r["includes_zero"] else "Baseline Better",
                axis=1
            )
            cols_show = ["Horizon", "best_baseline", "Δ MAE", "95% CI", "Conclusion"]
            st.dataframe(ci_disp[cols_show], hide_index=True, use_container_width=True)

    # Winter vs Non-Winter Split CI
    if "winter" in ci_dict:
        st.markdown("##### Winter vs. Non-Winter Subgroup Error Analysis")
        w_df = ci_dict["winter"].copy()
        w_df["Horizon"] = w_df["horizon"].apply(lambda h: f"Day +{h}")
        w_df["RF MAE"] = w_df["rf_mae"].round(1)
        w_df["Base MAE"] = w_df["baseline_mae"].round(1)
        w_df["95% CI"] = w_df.apply(lambda r: f"[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]", axis=1)
        w_disp = w_df[["Horizon", "season", "days_count", "RF MAE", "Base MAE", "95% CI"]].copy()
        w_disp.columns = ["Horizon", "Season", "Days", "RF MAE", "Base MAE", "95% Confidence Interval"]
        st.dataframe(w_disp, hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── Section D: Test Predictions Diagnostics ────────────────────────────────
    st.markdown("### D. Test Prediction Diagnostics & Residual Distributions")
    st.caption("Inspect residual scatter and error distributions for any chosen forecast horizon.")

    h_choice = st.selectbox("Select Horizon for Detailed Scatter & Residuals:", [1, 2, 3, 4, 5], format_func=lambda h: f"Horizon Day +{h}")

    c_scat, c_res = st.columns(2)
    with c_scat:
        fig_scat = create_actual_vs_predicted_scatter(test_preds, horizon=h_choice)
        st.plotly_chart(fig_scat, use_container_width=True)

    with c_res:
        fig_res = create_residuals_distribution_chart(test_preds, horizon=h_choice)
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # ── Section E: Classification & Alert Metrics ──────────────────────────────
    st.markdown("### E. CPCB Category Classification Performance")
    st.caption("Per-category Precision, Recall, and F1 score across the 6 India NAQI bands.")

    per_class_rf = per_class_df[(per_class_df["model"] == "Random Forest") & (per_class_df["horizon"] == h_choice)].copy()
    if len(per_class_rf) > 0:
        p_disp = per_class_rf[["category", "precision", "recall", "f1", "support"]].copy()
        p_disp.columns = ["Category", "Precision", "Recall", "F1 Score", "Test Support (Days)"]
        p_disp["Precision"] = p_disp["Precision"].round(3)
        p_disp["Recall"] = p_disp["Recall"].round(3)
        p_disp["F1 Score"] = p_disp["F1 Score"].round(3)
        st.dataframe(p_disp, hide_index=True, use_container_width=True)

    # ── Section F: Worst Prediction Days & Diwali Analysis ────────────────────
    if h_choice in worst_dict:
        st.markdown(f"### F. Worst Prediction Days for Horizon Day +{h_choice}")
        st.caption("Top 10 highest-error days on the test set, identifying severe episodes and proximity to Diwali.")

        w_df = worst_dict[h_choice].copy()
        st.dataframe(w_df, hide_index=True, use_container_width=True)
