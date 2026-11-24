"""
dashboard/pages/historical.py
Page 3: Historical Pollution Analysis
Interactive exploration of Delhi's 2018–2026 air pollution observations.
Provides multi-criteria filtering, temporal trends, particulate ratios,
seasonal dynamics, category distributions, correlation heatmaps, and dataset export.
"""

from datetime import date
import pandas as pd
import streamlit as st

from dashboard.components import render_header, render_freshness_banner
from dashboard.data_service import get_clean_daily, get_dataset_summary
from dashboard.charts import (
    create_historical_trend_chart,
    create_particulates_comparison_chart,
    create_monthly_seasonal_chart,
    create_category_distribution_chart,
    create_correlation_heatmap,
)

POLLUTANT_METADATA = {
    "aqi": ("Air Quality Index", "Index points (0–500)"),
    "pm25": ("Fine Particulate Matter (PM2.5)", "µg/m³"),
    "pm10": ("Coarse Particulate Matter (PM10)", "µg/m³"),
    "no2": ("Nitrogen Dioxide (NO2)", "µg/m³"),
    "so2": ("Sulfur Dioxide (SO2)", "µg/m³"),
    "co": ("Carbon Monoxide (CO)", "mg/m³"),
    "o3": ("Ozone (O3)", "µg/m³"),
    "nh3": ("Ammonia (NH3)", "µg/m³"),
}


def render():
    summary = get_dataset_summary()
    daily_df = get_clean_daily()

    render_header(
        title="Historical Pollution Analysis",
        subtitle="Empirical exploration of Delhi's spatial air quality records from 2018 to 2026.",
        badge="Data Exploration & Trends",
    )

    render_freshness_banner(summary["latest_date_str"])

    # ── Interactive Filter Controls ───────────────────────────────────────────
    st.markdown("### Filter Observations")
    fc1, fc2, fc3 = st.columns([1.5, 1.2, 1.3])

    with fc1:
        date_range = st.date_input(
            "Observation Date Range",
            value=[date(2023, 1, 1), summary["end_dt"].date()],
            min_value=summary["start_dt"].date(),
            max_value=summary["end_dt"].date(),
        )

    with fc2:
        pollutant_choice = st.selectbox(
            "Primary Pollutant / Metric",
            options=list(POLLUTANT_METADATA.keys()),
            format_func=lambda x: f"{x.upper()} — {POLLUTANT_METADATA[x][0]}",
            index=0,
        )

    with fc3:
        all_cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        chosen_cats = st.multiselect(
            "Filter AQI Categories",
            options=all_cats,
            default=all_cats,
        )

    # Filter dataframe
    if isinstance(date_range, list) and len(date_range) == 2:
        s_date, e_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        s_date, e_date = summary["start_dt"], summary["end_dt"]

    mask = (
        (daily_df["date"] >= s_date) &
        (daily_df["date"] <= e_date) &
        (daily_df["aqi_category"].isin(chosen_cats))
    )
    filtered_df = daily_df[mask].copy().sort_values("date").reset_index(drop=True)

    if len(filtered_df) == 0:
        st.warning("No observations match the selected filter criteria. Please broaden your selection.")
        return

    # Filter KPI summaries
    st.markdown(
        f"<div style='font-size: 0.82rem; color: #6C7770; margin-bottom: 0.8rem;'>"
        f"Displaying <strong>{len(filtered_df):,} days</strong> ({s_date.strftime('%b %d, %Y')} to {e_date.strftime('%b %d, %Y')}) "
        f"| Mean AQI: <strong>{filtered_df['aqi'].mean():.1f}</strong> | Median AQI: <strong>{filtered_df['aqi'].median():.1f}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── 1. Primary Time-Series Trend ──────────────────────────────────────────
    poll_name, poll_unit = POLLUTANT_METADATA[pollutant_choice]
    fig_trend = create_historical_trend_chart(filtered_df, pollutant_choice, poll_name, poll_unit)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── 2. Particulates Analysis & Monthly Seasonality ────────────────────────
    col_pm, col_seas = st.columns(2)

    with col_pm:
        fig_pm = create_particulates_comparison_chart(filtered_df)
        st.plotly_chart(fig_pm, use_container_width=True)

    with col_seas:
        fig_seas = create_monthly_seasonal_chart(daily_df)
        st.plotly_chart(fig_seas, use_container_width=True)

    st.markdown("---")

    # ── 3. Category Distribution & Correlation Matrix ────────────────────────
    col_dist, col_corr = st.columns([1, 1.2])

    with col_dist:
        fig_dist = create_category_distribution_chart(filtered_df)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_corr:
        fig_corr = create_correlation_heatmap(filtered_df)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # ── 4. Dataset Preview & CSV Export ──────────────────────────────────────
    st.markdown("### Filtered Dataset Table")
    st.caption("City-wide daily spatially aggregated pollutant averages across retained CPCB stations.")

    preview_cols = [
        "date", "aqi", "aqi_category", "prominent_pollutant",
        "pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "wind_speed"
    ]
    st.dataframe(filtered_df[preview_cols], use_container_width=True, hide_index=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Data as CSV ({len(filtered_df):,} records)",
        data=csv_data,
        file_name="delhi_aqi_historical_filtered.csv",
        mime="text/csv",
    )
