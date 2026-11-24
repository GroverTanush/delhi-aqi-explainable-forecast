"""
dashboard/styles.py
Custom styling and CSS injection for Delhi AQI Intelligence Dashboard.
Adheres strictly to the restrained, data-focused, minimal aesthetic:
- Palette:
  Background: #F6F7F4
  Card Background: #FFFFFF
  Primary Text: #202923
  Secondary Text: #6C7770
  Primary Accent: #246B57
  Secondary Accent: #9BB7A5
  Borders: #E3E8E2
"""

# Color Constants
BG_COLOR = "#F6F7F4"
CARD_BG = "#FFFFFF"
PRIMARY_TEXT = "#202923"
SECONDARY_TEXT = "#6C7770"
PRIMARY_ACCENT = "#246B57"
SECONDARY_ACCENT = "#9BB7A5"
BORDER_COLOR = "#E3E8E2"

# CPCB Category Palette (Muted, professional data tones)
AQI_CATEGORY_STYLES = {
    "Good": {
        "color": "#1B8755",
        "bg": "#EBF6F0",
        "border": "#A8D8B9",
        "desc": "Minimal impact on health"
    },
    "Satisfactory": {
        "color": "#4E7B24",
        "bg": "#F2F8EC",
        "border": "#BEDBA2",
        "desc": "Minor breathing discomfort to sensitive people"
    },
    "Moderate": {
        "color": "#A16207",
        "bg": "#FEFCE8",
        "border": "#FDE047",
        "desc": "Breathing discomfort to people with lung disease"
    },
    "Poor": {
        "color": "#C2410C",
        "bg": "#FFF7ED",
        "border": "#FDBA74",
        "desc": "Breathing discomfort to most people on prolonged exposure"
    },
    "Very Poor": {
        "color": "#B91C1C",
        "bg": "#FEF2F2",
        "border": "#FCA5A5",
        "desc": "Respiratory illness on prolonged exposure"
    },
    "Severe": {
        "color": "#6B21A8",
        "bg": "#FAF5FF",
        "border": "#D8B4FE",
        "desc": "Affects healthy people and seriously impacts those with existing diseases"
    },
}

CUSTOM_CSS = f"""
<style>
/* Main layout and background */
.stApp {{
    background-color: {BG_COLOR};
    color: {PRIMARY_TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background-color: #FFFFFF;
    border-right: 1px solid {BORDER_COLOR};
    padding-top: 1rem;
}}

/* Clean typography */
h1, h2, h3, h4, h5, h6 {{
    color: {PRIMARY_TEXT};
    font-weight: 600;
    letter-spacing: -0.01em;
}}

/* Header container */
.dash-header {{
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid {BORDER_COLOR};
}}

.dash-header-title {{
    font-size: 1.85rem;
    font-weight: 700;
    color: {PRIMARY_TEXT};
    margin: 0;
    line-height: 1.2;
}}

.dash-header-subtitle {{
    font-size: 0.95rem;
    color: {SECONDARY_TEXT};
    margin-top: 0.35rem;
    margin-bottom: 0.5rem;
}}

.dash-badge {{
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    background-color: #E8EFEA;
    color: {PRIMARY_ACCENT};
    border: 1px solid {SECONDARY_ACCENT};
}}

/* Data Freshness Warning Banner */
.freshness-banner {{
    background-color: #FAFBF9;
    border: 1px solid {BORDER_COLOR};
    border-left: 4px solid {PRIMARY_ACCENT};
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
    color: {PRIMARY_TEXT};
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.freshness-banner strong {{
    color: {PRIMARY_ACCENT};
}}

/* Cards */
.aqi-card {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 1.15rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}}

.aqi-card-header {{
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {SECONDARY_TEXT};
    margin-bottom: 0.4rem;
}}

.aqi-card-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {PRIMARY_TEXT};
    line-height: 1.1;
    margin-bottom: 0.25rem;
}}

.aqi-card-subtext {{
    font-size: 0.8rem;
    color: {SECONDARY_TEXT};
}}

/* Category Badge */
.category-pill {{
    display: inline-flex;
    align-items: center;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    margin-top: 0.25rem;
}}

/* Alert Box */
.alert-box {{
    background-color: #FFF7ED;
    border: 1px solid #FDBA74;
    border-left: 4px solid #C2410C;
    border-radius: 4px;
    padding: 0.85rem 1.15rem;
    margin-bottom: 1.25rem;
    color: #9A3412;
    font-size: 0.88rem;
}}

.alert-box-title {{
    font-weight: 600;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

/* Stat grid */
.stat-pill {{
    background-color: #FFFFFF;
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 0.75rem 1rem;
    text-align: center;
}}

.stat-pill-num {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {PRIMARY_ACCENT};
}}

.stat-pill-label {{
    font-size: 0.75rem;
    color: {SECONDARY_TEXT};
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-top: 0.15rem;
}}

/* Clean table presentation */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    background-color: #FFFFFF;
}}

/* Clean button styling */
.stButton > button {{
    background-color: {PRIMARY_ACCENT};
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.45rem 1.1rem;
    transition: background-color 0.15s ease;
}}

.stButton > button:hover {{
    background-color: #1b5343;
    color: #FFFFFF;
}}

.stDownloadButton > button {{
    background-color: #FFFFFF;
    color: {PRIMARY_TEXT};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.35rem 0.85rem;
}}

.stDownloadButton > button:hover {{
    border-color: {PRIMARY_ACCENT};
    color: {PRIMARY_ACCENT};
}}
</style>
"""


def get_category_style(category: str) -> dict:
    """Returns style dict for a given AQI category."""
    return AQI_CATEGORY_STYLES.get(category, {
        "color": PRIMARY_TEXT,
        "bg": "#F3F4F6",
        "border": BORDER_COLOR,
        "desc": ""
    })
