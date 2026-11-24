"""
dashboard/pages/methodology.py
Page 6: System Methodology & Architecture
Detailed end-to-end documentation of the data ingestion, quality control, NAQI calculation,
feature engineering, chronological splitting, multi-horizon models, statistical evaluation,
known system limitations, and future scope.
"""

import streamlit as st
from dashboard.components import render_header, render_freshness_banner
from dashboard.data_service import get_dataset_summary


def render():
    summary = get_dataset_summary()

    render_header(
        title="How the Forecasting System Works",
        subtitle="End-to-end methodology, feature engineering pipeline, validation standards, and architectural design.",
        badge="Engineering Architecture",
    )

    render_freshness_banner(summary["latest_date_str"])

    # ── 1. End-to-End Pipeline Visualization ─────────────────────────────────
    st.markdown("### 1. End-to-End Data & ML Pipeline")
    st.caption("A fully reproducible chronological architecture designed to completely eliminate data leakage.")

    pipeline_steps = [
        ("1. CPCB Stations", "37 continuous monitoring stations across Delhi logging hourly pollutant & weather telemetry (2018–2026)."),
        ("2. Data Cleaning", "Filtering stations with <50% coverage; converting mixed barometric pressure (mmHg to hPa); removing stuck sensors (>=24h frozen readings); filtering particulate inversions (PM2.5 > 1.2 * PM10)."),
        ("3. City-Wide Aggregation", "Spatial hourly averaging across >=3 reporting stations, followed by 24h daily aggregation (>=16 valid hours required for criteria pollutants)."),
        ("4. CPCB NAQI Computation", "Calculating sub-indices for 7 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3, NH3) using CPCB 2014 breakpoints. Prominent pollutant determined by max sub-index; final AQI clipped to [0, 500]."),
        ("5. Feature Engineering", "Constructing 38 strictly historical features using trailing-only windows: AQI lags (1, 2, 3, 7d), 3d/7d rolling means, pollutant rolling means & daily maxes, meteorological variables, and cyclical calendar features."),
        ("6. Chronological Splitting", "Train (2018-01-08 to 2023-03-31; 1,887 days), Val (2023-04-06 to 2024-03-31; 352 days), and Test (2024-04-06 to 2026-09-25; 887 days). A 5-day buffer gap is enforced between partitions to prevent target leakage."),
        ("7. Multi-Horizon Models", "Direct multi-step forecasting: 5 separate Random Forest regressors (rf_h1.pkl to rf_h5.pkl) tuned on Train+Val, compared against Ridge and 3 classical baselines."),
        ("8. Statistical Evaluation", "Evaluating regression (MAE, RMSE, R²), classification (6-class accuracy, F1, alert metrics), and paired moving-block bootstrap confidence intervals (B=2,000, 7-day blocks)."),
        ("9. SHAP Explainability", "Interpreting tree ensembles with shap.TreeExplainer: decomposing attribution into 4 thematic families and generating instance waterfall plots."),
    ]

    for title, desc in pipeline_steps:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E3E8E2; border-left: 3px solid #246B57; border-radius: 4px; padding: 0.65rem 0.95rem; margin-bottom: 0.55rem;">
                <div style="font-weight: 600; font-size: 0.88rem; color: #202923; margin-bottom: 0.15rem;">{title}</div>
                <div style="font-size: 0.82rem; color: #4A554E; line-height: 1.4;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 2. Feature Engineering Breakdown ──────────────────────────────────────
    st.markdown("### 2. Feature Architecture (38 Engineered Features)")
    st.caption("All features are computed using strictly trailing observations available up to base date t.")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown(
            """
            <div class="aqi-card">
                <div class="aqi-card-header" style="color: #246B57;">AQI History (9 Features)</div>
                <div style="font-size: 0.82rem; color: #4A554E; line-height: 1.5;">
                    • <code>aqi</code>: Today's city-wide AQI<br>
                    • <code>aqi_lag1, aqi_lag2, aqi_lag3, aqi_lag7</code>: Autoregressive lags<br>
                    • <code>aqi_rolling_mean_3d</code>: 3-day trailing mean (min_periods=3)<br>
                    • <code>aqi_rolling_mean_7d</code>: 7-day trailing mean (min_periods=5)<br>
                    • <code>aqi_rolling_max_7d</code>: 7-day trailing peak AQI<br>
                    • <code>aqi_trend_3d</code>: 3-day momentum (aqi - aqi_lag3)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="aqi-card">
                <div class="aqi-card-header" style="color: #C2410C;">Pollutant Telemetry (14 Features)</div>
                <div style="font-size: 0.82rem; color: #4A554E; line-height: 1.5;">
                    • Criteria concentrations: <code>pm25, pm10, no2, so2, co, o3</code><br>
                    • 3-day trailing rolling means for all 6 criteria pollutants<br>
                    • Peak exposure metrics: <code>pm25_max, pm10_max</code> (daily maximum 1-hour concentration across stations)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_f2:
        st.markdown(
            """
            <div class="aqi-card">
                <div class="aqi-card-header" style="color: #D97706;">Meteorology & Dispersion (9 Features)</div>
                <div style="font-size: 0.82rem; color: #4A554E; line-height: 1.5;">
                    • Instantaneous weather: <code>temperature, humidity, wind_speed, barometric_pressure</code> (hPa)<br>
                    • Wind direction circular components: <code>wind_dir_sin, wind_dir_cos</code><br>
                    • Trailing smoothed weather: 3-day rolling means of temperature, humidity, and wind speed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="aqi-card">
                <div class="aqi-card-header" style="color: #2E7D32;">Calendar & Seasonality (6 Features)</div>
                <div style="font-size: 0.82rem; color: #4A554E; line-height: 1.5;">
                    • <code>month, day_of_week, is_weekend</code><br>
                    • Cyclical day-of-year encoding: <code>day_of_year_sin, day_of_year_cos</code><br>
                    • <code>stubble_season</code>: Binary indicator flag for peak northern stubble burning window (Oct 15 to Nov 15)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 3. Rigorous Splitting & Leakage Prevention ────────────────────────────
    st.markdown("### 3. Chronological Time-Series Validation (Rule 2)")
    st.markdown(
        """
        Standard k-fold cross-validation is completely invalid for time-series forecasting because random shuffling leaks 
        future seasonal information into the past. This project enforces strict chronological splitting:
        """
    )

    split_cols = st.columns(3)
    with split_cols[0]:
        st.markdown(
            """
            <div class="stat-pill" style="text-align: left;">
                <div style="font-weight: 600; color: #202923;">Train Partition</div>
                <div style="font-size: 0.8rem; color: #6C7770; margin-top: 0.2rem;">2018-01-08 to 2023-03-31</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #246B57; margin-top: 0.4rem;">1,887 days</div>
                <div style="font-size: 0.72rem; color: #6C7770; margin-top: 0.2rem;">Initial model fitting & CV</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with split_cols[1]:
        st.markdown(
            """
            <div class="stat-pill" style="text-align: left;">
                <div style="font-weight: 600; color: #202923;">Validation Partition</div>
                <div style="font-size: 0.8rem; color: #6C7770; margin-top: 0.2rem;">2023-04-06 to 2024-03-31</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #246B57; margin-top: 0.4rem;">352 days</div>
                <div style="font-size: 0.72rem; color: #6C7770; margin-top: 0.2rem;">Hyperparameter tuning (5-day buffer)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with split_cols[2]:
        st.markdown(
            """
            <div class="stat-pill" style="text-align: left;">
                <div style="font-weight: 600; color: #202923;">Test Partition</div>
                <div style="font-size: 0.8rem; color: #6C7770; margin-top: 0.2rem;">2024-04-06 to 2026-09-25</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #246B57; margin-top: 0.4rem;">887 days</div>
                <div style="font-size: 0.72rem; color: #6C7770; margin-top: 0.2rem;">Untouched holdout evaluation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("A 5-day buffer gap is dropped between train/val and val/test to prevent multi-day target overlap from leaking across partition boundaries.")

    st.markdown("---")

    # ── 4. Limitations & Future Scope ─────────────────────────────────────────
    c_lim, c_fut = st.columns(2)

    with c_lim:
        st.markdown("### 4. Known System Limitations")
        st.markdown(
            """
            <div class="aqi-card" style="font-size: 0.82rem; color: #4A554E; line-height: 1.55;">
                • <strong>City-Wide Spatial Averaging:</strong> The models forecast the Delhi city-wide daily mean AQI across 35 stations. Microclimate variations and localized street-canyon hotspots (e.g., Anand Vihar vs Mandir Marg) are smoothed out.<br><br>
                • <strong>No Numerical Future Weather Forecasts (NWP):</strong> Features only use weather observations up to day t. Sudden meteorological frontal passages, unexpected rain, or sudden wind shear after day t are unknown to the model.<br><br>
                • <strong>CPCB NAQI Ceiling Cap (500 AQI):</strong> Official CPCB definitions saturate at 500. Severe catastrophic episodes exceeding 500 are clipped, attenuating peak regression predictions.<br><br>
                • <strong>Horizon Skill Decay:</strong> Beyond Day +1, regression errors increase significantly, approaching seasonal persistence baselines at Day +4 and +5.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_fut:
        st.markdown("### 5. Future Scope & Extensions")
        st.markdown(
            """
            <div class="aqi-card" style="font-size: 0.82rem; color: #4A554E; line-height: 1.55;">
                • <strong>NWP Weather Integration:</strong> Ingesting 5-day forward numerical weather forecasts from IMD / ECMWF (boundary layer height, forecast wind vectors, ventilation index) to improve long-horizon skill.<br><br>
                • <strong>Satellite Stubble Fire Ingestion:</strong> Coupling real-time VIIRS / MODIS thermal anomaly fire counts from Punjab and Haryana during the Oct 15–Nov 15 agricultural burning window.<br><br>
                • <strong>Probabilistic Conformal Prediction:</strong> Generating distribution-free, guaranteed valid prediction intervals instead of point forecasts with static historical MAE.<br><br>
                • <strong>Spatial Downscaling:</strong> Training multi-task neural or hierarchical spatial models to downscale city forecasts to individual monitoring stations.
            </div>
            """,
            unsafe_allow_html=True,
        )
