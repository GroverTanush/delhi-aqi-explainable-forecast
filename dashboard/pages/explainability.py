"""
dashboard/pages/explainability.py
Page 4: Explainable AI / SHAP Attribution
Provides in-depth model interpretability using SHapley Additive exPlanations (TreeExplainer).
Covers feature family transitions across horizons, global beeswarm summaries, bar importance rankings,
and case-study waterfall decompositions on Severe winter test episodes.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.components import render_header, render_freshness_banner
from dashboard.data_service import get_shap_family_importance, get_dataset_summary
from dashboard.charts import create_shap_family_bar

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CASE_STUDY_METADATA = {
    ("h1", "good"): {
        "title": "Horizon h=1 | Smallest Error Case (2025-12-26 → 2025-12-27)",
        "actual": 407.4,
        "predicted": 402.3,
        "error": 5.09,
        "file": "shap_waterfall_h1_good.png",
        "desc": (
            "Severe winter day forecast with exceptional precision. High base particulates "
            "(pm25: 215.1, pm25_max: 273.7) provide massive positive pushes (+58.7 and +34.2 AQI points) "
            "above the baseline expectation, accurately projecting the continued Severe category state."
        ),
    },
    ("h1", "bad"): {
        "title": "Horizon h=1 | Largest Error Case (2024-11-12 → 2024-11-13)",
        "actual": 500.0,
        "predicted": 391.7,
        "error": 108.27,
        "file": "shap_waterfall_h1_bad.png",
        "desc": (
            "Severe post-Diwali episode where the ground truth hit the maximum CPCB NAQI cap (500.0). "
            "While the model correctly predicted a Very Poor/Severe spike (391.7), it underpredicted the absolute peak "
            "by 108 points due to extreme localized meteorological stagnancy not captured in daily trailing means."
        ),
    },
    ("h5", "good"): {
        "title": "Horizon h=5 | Smallest Error Case (2025-11-03 → 2025-11-08)",
        "actual": 403.4,
        "predicted": 402.3,
        "error": 1.04,
        "file": "shap_waterfall_h5_good.png",
        "desc": (
            "5-day forward prediction targeting Severe winter conditions with under 1.1 AQI error. "
            "The seasonal cyclical feature day_of_year_cos (+61.4 points) and 3-day particulate rolling means "
            "accurately pushed the 5-day trajectory into the Severe band without relying on daily transient noise."
        ),
    },
    ("h5", "bad"): {
        "title": "Horizon h=5 | Largest Error Case (2025-10-16 → 2025-10-21)",
        "actual": 454.7,
        "predicted": 275.9,
        "error": 178.72,
        "file": "shap_waterfall_h5_bad.png",
        "desc": (
            "5-day forecast during the onset of the stubble-burning and pre-Diwali pollution surge. "
            "On October 16, daily pollution was still Moderate/Poor; the 5-day forward Random Forest lacked numerical "
            "weather forecasts (NWP) or fire-count telemetry, underestimating the sudden catastrophic inversion on October 21."
        ),
    },
}


def render():
    summary = get_dataset_summary()
    family_df = get_shap_family_importance()

    render_header(
        title="Understanding the Model's Decisions",
        subtitle="Explore which historical features influence the AQI forecasts using TreeExplainer SHAP attribution.",
        badge="Model Interpretability",
    )

    render_freshness_banner(summary["latest_date_str"])

    # Educational Context Box
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid #E3E8E2; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
            <div style="font-weight: 600; font-size: 0.95rem; color: #202923; margin-bottom: 0.35rem;">
                Methodological Note: How SHAP Interprets Random Forests
            </div>
            <div style="font-size: 0.85rem; color: #4A554E; line-height: 1.5;">
                <strong>SHAP (SHapley Additive exPlanations)</strong> calculates the cooperative game-theoretic marginal contribution of each feature 
                to a specific forecast. For any prediction: 
                <br>
                <code style="background-color: #F1F4F2; padding: 2px 6px; border-radius: 4px; color: #246B57;">
                    Prediction = Base Value (Dataset Mean) + ∑(SHAP Contributions)
                </code>
                <br><br>
                <strong>Credit Splitting Across Correlated Features:</strong> Mutual correlation among features 
                (e.g., <code>pm25</code>, <code>pm25_rolling_mean_3d</code>, <code>aqi</code>, <code>pm25_max</code>) causes tree ensembles 
                to partition attribution across these related variables. To understand the underlying physical signal, features are 
                grouped into 4 thematic families: <strong>Pollutants, AQI history, Calendar, and Weather</strong>.
                <br>
                <em>Note: SHAP demonstrates model dependency, not atmospheric causality.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section A: Feature Family Importance Across Horizons ──────────────────
    st.markdown("### A. Feature Family Importance Across Horizons")
    st.caption("How model attribution dynamically transitions from daily pollutant readings to seasonal calendar cycles as the forecast horizon expands.")

    c_mode, _ = st.columns([1.5, 2.5])
    with c_mode:
        view_mode = st.radio(
            "Display Metric:",
            ["absolute", "percentage"],
            format_func=lambda x: "Total Mean |SHAP| (AQI Points)" if x == "absolute" else "Percentage Share (%)",
            horizontal=True,
        )

    fig_fam = create_shap_family_bar(family_df, mode=view_mode)
    st.plotly_chart(fig_fam, use_container_width=True)

    # Family Table
    col_tbl, col_takeaway = st.columns([1.3, 1])

    with col_tbl:
        st.markdown("##### Detailed Family Attribution by Horizon")
        fam_disp = family_df.copy()
        fam_disp.columns = ["Horizon", "Feature Family", "Mean |SHAP| (AQI)", "Share of Total (%)"]
        fam_disp["Horizon"] = fam_disp["Horizon"].apply(lambda h: f"Day +{h}")
        fam_disp["Mean |SHAP| (AQI)"] = fam_disp["Mean |SHAP| (AQI)"].round(2)
        fam_disp["Share of Total (%)"] = fam_disp["Share of Total (%)"].round(1).apply(lambda p: f"{p}%")
        st.dataframe(fam_disp, hide_index=True, use_container_width=True)

    with col_takeaway:
        st.markdown(
            """
            <div class="aqi-card" style="height: 100%;">
                <div class="aqi-card-header">Scientific Findings</div>
                <ul style="font-size: 0.82rem; color: #4A554E; padding-left: 1.1rem; line-height: 1.5; margin-bottom: 0;">
                    <li><strong>Day +1 Horizon:</strong> Pollutants (61.6%) and AQI history (28.7%) account for <strong>90.3%</strong> of total attribution. Short-term forecasting is dominated by current particulate load.</li>
                    <li><strong>Day +3 Horizon:</strong> Trailing multi-day averages become more important than instantaneous readings, and calendar seasonality rises to <strong>19.8%</strong>.</li>
                    <li><strong>Day +5 Horizon:</strong> Calendar features surge to <strong>30.7%</strong> of attribution, while instantaneous pollutant importance declines. 5-day forecasts are primarily governed by seasonal position.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Section B: Global SHAP Beeswarm & Feature Importance Plots ─────────────
    st.markdown("### B. Global Feature Importance (Beeswarm & Mean |SHAP|)")
    st.caption("Inspect the top 15 features governing model decisions across Horizons 1, 3, and 5.")

    chosen_horizon = st.selectbox(
        "Select Forecast Horizon for Global Plots:",
        [1, 3, 5],
        format_func=lambda h: f"Horizon Day +{h} (rf_h{h}.pkl)",
    )

    out_dir = REPO_ROOT / "outputs"
    beeswarm_path = out_dir / f"shap_summary_h{chosen_horizon}.png"
    bar_path = out_dir / f"shap_bar_h{chosen_horizon}.png"

    c_bee, c_bar = st.columns(2)

    with c_bee:
        st.markdown(f"##### Beeswarm Summary Plot (h={chosen_horizon})")
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), use_container_width=True)
            st.caption("Red dots = high feature value; Blue dots = low feature value. Positive SHAP pushes AQI higher.")
        else:
            st.info(f"Beeswarm plot for h={chosen_horizon} not found.")

    with c_bar:
        st.markdown(f"##### Mean Absolute |SHAP| Importance (h={chosen_horizon})")
        if bar_path.exists():
            st.image(str(bar_path), use_container_width=True)
            st.caption("Ranked by average magnitude of impact across the 887 test evaluation days.")
        else:
            st.info(f"Bar plot for h={chosen_horizon} not found.")

    st.markdown("---")

    # ── Section C: Severe Winter Days Waterfall Case Studies ───────────────────
    st.markdown("### C. Severe Winter Day Case Studies (Waterfall Decompositions)")
    st.caption("Detailed instance-level waterfall explanations for the best and worst predictions on Severe winter test days (truth > 400 AQI).")

    col_h_choice, col_case_choice = st.columns([1, 1])

    with col_h_choice:
        wf_h = st.selectbox("Select Case Horizon:", ["h1", "h5"], format_func=lambda x: "Horizon Day +1" if x == "h1" else "Horizon Day +5")

    with col_case_choice:
        wf_case = st.selectbox("Select Case Study:", ["good", "bad"], format_func=lambda x: "Smallest Error Case (Accurate Prediction)" if x == "good" else "Largest Error Case (Underprediction Failure Mode)")

    case_meta = CASE_STUDY_METADATA.get((wf_h, wf_case))

    if case_meta:
        st.markdown(
            f"""
            <div class="aqi-card" style="border-left: 4px solid {'#1B8755' if wf_case == 'good' else '#B91C1C'};">
                <div style="font-weight: 600; font-size: 0.95rem; color: #202923; margin-bottom: 0.25rem;">
                    {case_meta['title']}
                </div>
                <div style="font-size: 0.82rem; color: #6C7770; margin-bottom: 0.4rem;">
                    Observed Truth: <strong>{case_meta['actual']} AQI</strong> | Model Prediction: <strong>{case_meta['predicted']} AQI</strong> | Absolute Error: <strong>{case_meta['error']} points</strong>
                </div>
                <div style="font-size: 0.85rem; color: #4A554E; line-height: 1.45;">
                    {case_meta['desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        wf_img_path = out_dir / case_meta["file"]
        if wf_img_path.exists():
            st.image(str(wf_img_path), use_container_width=True)
            st.caption("SHAP waterfall plot decomposing the instance prediction starting from E[f(x)] base value to final f(x).")
        else:
            st.info(f"Waterfall image {case_meta['file']} not found.")
