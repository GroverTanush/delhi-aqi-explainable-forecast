"""
dashboard/pages/forecast.py
Page 2: Forecast Explorer
Interactive multi-horizon AQI forecast generator using actual trained Random Forest models.
Allows historical date selection, generates Day +1 to +5 predictions, compares against subsequent actuals,
provides interactive error analysis, CSV download, and dynamic TreeExplainer SHAP attribution.
"""

from datetime import date
import pandas as pd
import streamlit as st

from dashboard.components import (
    render_header,
    render_freshness_banner,
    render_category_pill,
    render_forecast_card,
    render_alert_box,
)
from dashboard.data_service import get_dataset_summary, get_clean_daily
from dashboard.prediction_service import generate_forecast, explain_single_prediction
from dashboard.charts import create_forecast_figure


def render():
    summary = get_dataset_summary()
    daily_df = get_clean_daily()

    render_header(
        title="Forecast Explorer",
        subtitle="Simulate and verify five-day forward AQI forecasts using trained Random Forest regressors.",
        badge="Inference & Validation",
    )

    render_freshness_banner(summary["latest_date_str"])

    # Date Selection Controls
    st.markdown("### Select Forecast Base Date")
    st.caption("Features are computed strictly using observations up to and including the base date (no future leakage). Requires >= 7 trailing days.")

    min_date = date(2018, 1, 8)
    max_date = date(2026, 9, 30)

    # Session state for selected date
    if "forecast_date" not in st.session_state:
        st.session_state.forecast_date = max_date

    c_date, c_quick, c_btn = st.columns([1.5, 1, 1])

    with c_date:
        selected_date = st.date_input(
            "Forecast Base Date (t)",
            value=st.session_state.forecast_date,
            min_value=min_date,
            max_value=max_date,
            help="Select any date between 2018-01-08 and 2026-09-30."
        )

    with c_quick:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button("Latest Date (2026-09-30)", use_container_width=True):
            st.session_state.forecast_date = max_date
            st.rerun()

    with c_btn:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        generate_clicked = st.button("Generate 5-Day Forecast", type="primary", use_container_width=True)

    # Ensure date string
    sel_date_str = selected_date.strftime("%Y-%m-%d")

    # Run Prediction
    with st.spinner(f"Computing 38 trailing features and running 5 Random Forest models for {sel_date_str}..."):
        try:
            forecast_df, feat_row, any_alert = generate_forecast(sel_date_str)
        except Exception as e:
            st.error(f"Prediction failed for date {sel_date_str}: {str(e)}")
            return

    st.markdown("---")

    # 1. High Pollution Episode Alert
    if any_alert:
        alert_days = [f"{r['horizon']} ({r['target_date']})" for _, r in forecast_df.iterrows() if r["predicted_aqi"] > 200]
        render_alert_box(
            title=f"High Pollution Episode Forecasted on {len(alert_days)} Horizon(s)",
            message=f"Model forecasts AQI > 200 (Poor or worse) on: <strong>{', '.join(alert_days)}</strong>. "
                    f"Sensory irritation and respiratory stress are likely in sensitive populations.",
            is_warning=True,
        )

    # 2. Forecast Cards Grid
    st.markdown(f"#### Predicted Forecast: Base Date {sel_date_str}")
    fcols = st.columns(5)
    for i, (_, row) in enumerate(forecast_df.iterrows()):
        with fcols[i]:
            render_forecast_card(row.to_dict())

    # 3. Interactive Trajectory Chart
    t_dt = pd.to_datetime(sel_date_str)
    fig = create_forecast_figure(daily_df, t_dt, forecast_df)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Error Analysis Table (Predicted vs Actual)
    st.markdown("#### Forecast Summary & Ground Truth Comparison")
    st.caption("Where ground truth exists in the dataset, actual AQI and absolute errors are computed for direct verification.")

    display_cols = [
        "horizon", "target_date", "predicted_aqi", "category",
        "typical_mae", "actual_aqi", "abs_error", "advisory"
    ]
    table_df = forecast_df[display_cols].copy()
    table_df.columns = [
        "Horizon", "Target Date", "Predicted AQI", "Category",
        "Historical MAE", "Observed AQI", "Absolute Error", "Advisory"
    ]

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # CSV Download Button
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Forecast as CSV (forecast_{sel_date_str}.csv)",
        data=csv_bytes,
        file_name=f"delhi_aqi_forecast_{sel_date_str}.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # 5. Dynamic SHAP Attribution for This Generated Forecast
    st.markdown("### Model Decision Attribution (SHAP TreeExplainer)")
    st.caption("Real-time SHAP explanation computed directly on the 38-feature vector of this forecast instance.")

    c_h, _ = st.columns([1.5, 3])
    with c_h:
        chosen_h = st.selectbox("Select Forecast Horizon to Explain:", [1, 3, 5], format_func=lambda x: f"Day +{x} Forecast")

    with st.spinner(f"Computing TreeExplainer SHAP values for Horizon {chosen_h}..."):
        try:
            shap_result = explain_single_prediction(sel_date_str, horizon=chosen_h)
            
            scol_base, scol_pred = st.columns(2)
            with scol_base:
                st.markdown(
                    f"""
                    <div class="aqi-card">
                        <div class="aqi-card-header">Model Base Value (Expected Value)</div>
                        <div class="aqi-card-value">{shap_result['base_value']:.1f} <span style="font-size: 0.85rem; color: #6C7770;">AQI</span></div>
                        <div class="aqi-card-subtext">Mean prediction across training set</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with scol_pred:
                st.markdown(
                    f"""
                    <div class="aqi-card">
                        <div class="aqi-card-header">Instance Model Prediction (Raw Target)</div>
                        <div class="aqi-card-value" style="color: #246B57;">{shap_result['prediction']:.1f} <span style="font-size: 0.85rem; color: #6C7770;">AQI</span></div>
                        <div class="aqi-card-subtext">Base Value + Sum of All SHAP Attributions</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            c_pos, c_neg = st.columns(2)
            with c_pos:
                st.markdown("##### 🔺 Top Features Pushing AQI Higher (Worse Air Quality)")
                pos_df = pd.DataFrame(shap_result["top_positive"])
                if len(pos_df) > 0:
                    pos_df["shap_value"] = pos_df["shap_value"].apply(lambda v: f"+{v:.2f}")
                    pos_df["feature_value"] = pos_df["feature_value"].apply(lambda v: f"{v:.2f}")
                    pos_df.columns = ["Feature", "Attribution (+AQI)", "Observed Value"]
                    st.dataframe(pos_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No positive attributions for this instance.")

            with c_neg:
                st.markdown("##### 🔻 Top Features Pushing AQI Lower (Better Air Quality)")
                neg_df = pd.DataFrame(shap_result["top_negative"])
                if len(neg_df) > 0:
                    neg_df["shap_value"] = neg_df["shap_value"].apply(lambda v: f"{v:.2f}")
                    neg_df["feature_value"] = neg_df["feature_value"].apply(lambda v: f"{v:.2f}")
                    neg_df.columns = ["Feature", "Attribution (-AQI)", "Observed Value"]
                    st.dataframe(neg_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No negative attributions for this instance.")

        except Exception as e:
            st.warning(f"Dynamic SHAP attribution unavailable: {str(e)}")
