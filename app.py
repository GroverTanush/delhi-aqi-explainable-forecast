"""
app.py
Delhi AQI Intelligence Dashboard
Main entry point for Streamlit application.
Coordinates routing, custom styling injection, and page rendering.
"""

import sys
from pathlib import Path
import streamlit as st

# Anchor repository root in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.styles import CUSTOM_CSS
from dashboard.data_service import get_dataset_summary
from dashboard.components import render_sidebar
from dashboard.pages import (
    overview,
    forecast,
    historical,
    explainability,
    evaluation,
    methodology,
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Delhi AQI Intelligence | Explainable Forecasting",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Global Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    try:
        summary = get_dataset_summary()
    except Exception as e:
        st.error(f"Critical error loading dataset metadata: {str(e)}")
        st.stop()

    # Render Custom Sidebar Navigation
    selected_page = render_sidebar(summary)

    # Route to Selected Page
    if selected_page == "Overview":
        overview.render()
    elif selected_page == "Forecast Explorer":
        forecast.render()
    elif selected_page == "Historical Analysis":
        historical.render()
    elif selected_page == "Explainable AI":
        explainability.render()
    elif selected_page == "Model Evaluation":
        evaluation.render()
    elif selected_page == "Methodology":
        methodology.render()
    else:
        overview.render()


if __name__ == "__main__":
    main()
