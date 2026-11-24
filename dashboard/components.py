"""
dashboard/components.py
Reusable UI components for Delhi AQI Intelligence Dashboard.
Adheres strictly to the minimal, clean, research-grade design standard.
"""

import streamlit as st
from dashboard.styles import (
    get_category_style,
    PRIMARY_TEXT,
    SECONDARY_TEXT,
    PRIMARY_ACCENT,
    BORDER_COLOR,
    CARD_BG,
)


def render_header(title: str, subtitle: str, badge: str | None = None):
    """Renders page header with clean typography and optional research badge."""
    badge_html = f'<span class="dash-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="dash-header">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                <h1 class="dash-header-title">{title}</h1>
                {badge_html}
            </div>
            <div class="dash-header-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_freshness_banner(latest_date_str: str):
    """Renders prominent data freshness disclaimer banner."""
    st.markdown(
        f"""
        <div class="freshness-banner">
            <div>
                <strong>Historical Research Dataset:</strong> Latest available observation is <strong>{latest_date_str}</strong>. 
                This application forecasts forward from historical observations and does not stream live CPCB sensor telemetry.
            </div>
            <div style="font-size: 0.75rem; color: #6C7770; white-space: nowrap; margin-left: 1rem;">
                35 Monitoring Stations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_category_pill(category: str, aqi_val: float | None = None) -> str:
    """Generates HTML category pill with distinct color, background and border."""
    style = get_category_style(category)
    label = f"{category} ({aqi_val:.0f})" if aqi_val is not None else category
    return (
        f'<span class="category-pill" style="color: {style["color"]}; '
        f'background-color: {style["bg"]}; border: 1px solid {style["border"]};">'
        f'{label}'
        f'</span>'
    )


def render_alert_box(title: str, message: str, is_warning: bool = True):
    """Renders prominent alert banner for severe pollution or data notice."""
    border_col = "#C2410C" if is_warning else PRIMARY_ACCENT
    bg_col = "#FFF7ED" if is_warning else "#F2F7F4"
    text_col = "#9A3412" if is_warning else "#1E473B"

    st.markdown(
        f"""
        <div class="alert-box" style="border-left-color: {border_col}; background-color: {bg_col}; color: {text_col};">
            <div class="alert-box-title">
                <span>⚠️</span> <span>{title}</span>
            </div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_card(r: dict):
    """Renders a single forecast card for Day +1 to Day +5."""
    cat = r["category"]
    style = get_category_style(cat)
    pill_html = render_category_pill(cat)

    actual_str = r.get("actual_aqi", "Unavailable")
    err_str = f"| Error: {r['abs_error']:.1f}" if r.get("has_actual") and not np.isnan(r.get("abs_error", np.nan)) else ""
    actual_subtext = f"Actual: {actual_str} {err_str}" if r.get("has_actual") else "Actual: Future Date"

    note_html = f'<div style="font-size: 0.7rem; color: #8F9A93; margin-top: 0.35rem; font-style: italic;">* {r["note"]}</div>' if r.get("note") else ""

    st.markdown(
        f"""
        <div class="aqi-card" style="border-top: 3px solid {style['color']};">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div class="aqi-card-header">{r['horizon']}</div>
                <div style="font-size: 0.75rem; color: #6C7770;">{r['target_date']}</div>
            </div>
            <div class="aqi-card-value">{r['predicted_aqi']:.0f} <span style="font-size: 0.85rem; font-weight: normal; color: #6C7770;">AQI</span></div>
            <div>{pill_html}</div>
            <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #6C7770;">
                <div>Typical test error: <strong>±{r['typical_mae']:.1f} MAE</strong></div>
                <div style="margin-top: 0.15rem; color: #4A554E;">{actual_subtext}</div>
            </div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
import numpy as np


def render_sidebar(summary: dict) -> str:
    """Renders custom sidebar navigation and returns selected page name."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding-bottom: 0.75rem; margin-bottom: 0.75rem; border-bottom: 1px solid {BORDER_COLOR};">
                <div style="font-size: 1.15rem; font-weight: 700; color: {PRIMARY_TEXT}; letter-spacing: -0.02em;">
                    Delhi AQI Intelligence
                </div>
                <div style="font-size: 0.78rem; color: {SECONDARY_TEXT}; margin-top: 0.15rem;">
                    Explainable Air Quality Forecasting
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_options = [
            "Overview",
            "Forecast Explorer",
            "Historical Analysis",
            "Explainable AI",
            "Model Evaluation",
            "Methodology",
        ]

        icons = ["📊", "🔮", "📈", "🧠", "🎯", "🔬"]
        labels = [f"{icon}  {name}" for icon, name in zip(icons, nav_options)]

        selected_label = st.radio(
            "Navigation",
            options=labels,
            index=0,
            label_visibility="collapsed",
        )
        selected_page = nav_options[labels.index(selected_label)]

        st.markdown(
            f"""
            <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid {BORDER_COLOR}; font-size: 0.75rem; color: {SECONDARY_TEXT};">
                <div style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: {PRIMARY_TEXT}; margin-bottom: 0.45rem;">
                    System Status
                </div>
                <div style="margin-bottom: 0.25rem;">
                    • Latest Date: <strong style="color: {PRIMARY_TEXT};">{summary['latest_date_str']}</strong>
                </div>
                <div style="margin-bottom: 0.25rem;">
                    • Coverage: <strong style="color: {PRIMARY_TEXT};">2018–2026</strong> ({summary['total_days']:,} days)
                </div>
                <div style="margin-bottom: 0.25rem;">
                    • CPCB Stations: <strong style="color: {PRIMARY_TEXT};">{summary['num_stations']} retained</strong>
                </div>
                <div style="margin-bottom: 0.25rem;">
                    • Models: <strong style="color: {PRIMARY_TEXT};">5 Random Forests</strong> (h=1..5)
                </div>
                <div style="margin-bottom: 0.25rem;">
                    • Features: <strong style="color: {PRIMARY_TEXT};">{summary['num_features']} strictly historical</strong>
                </div>
                <div style="margin-top: 1rem; padding-top: 0.75rem; border-top: 1px dashed {BORDER_COLOR}; font-size: 0.7rem; color: #8F9A93;">
                    Third-Year B.Tech CSE Capstone Project<br>
                    <a href="https://github.com/GroverTanush/delhi-aqi-explainable-forecast" target="_blank" style="color: {PRIMARY_ACCENT}; text-decoration: none;">GitHub Repository ↗</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected_page
