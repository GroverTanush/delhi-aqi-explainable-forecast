"""
dashboard/charts.py
Interactive Plotly charts for Delhi AQI Intelligence Dashboard.
Adheres strictly to the restrained, data-focused, minimal aesthetic.
Palette:
- Background: #F6F7F4
- Card Background: #FFFFFF
- Primary Text: #202923
- Secondary Text: #6C7770
- Primary Accent: #246B57
- Secondary Accent: #9BB7A5
- Borders: #E3E8E2
"""

from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Theme Colors
BG_COLOR = "#F6F7F4"
CARD_BG = "#FFFFFF"
PRIMARY_TEXT = "#202923"
SECONDARY_TEXT = "#6C7770"
PRIMARY_ACCENT = "#246B57"
SECONDARY_ACCENT = "#9BB7A5"
BORDER_COLOR = "#E3E8E2"

# Category Colors
CATEGORY_COLORS = {
    "Good": "#1B8755",
    "Satisfactory": "#558B2F",
    "Moderate": "#B78103",
    "Poor": "#C2410C",
    "Very Poor": "#B91C1C",
    "Severe": "#6B21A8",
}

FAMILY_COLORS = {
    "Pollutants": "#C2410C",
    "AQI history": "#246B57",
    "Calendar": "#2E7D32",
    "Weather": "#D97706",
}


def apply_clean_layout(fig: go.Figure, height: int = 400, title: str | None = None) -> go.Figure:
    """Applies clean, research-grade formatting to Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color=PRIMARY_TEXT, family="sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor="#FAFBFA",
        margin=dict(l=40, r=30, t=50 if title else 25, b=40),
        height=height,
        font=dict(family="sans-serif", color=PRIMARY_TEXT, size=11),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=12,
            font_family="sans-serif",
            bordercolor=BORDER_COLOR
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color=SECONDARY_TEXT),
            bgcolor="rgba(255, 255, 255, 0.7)",
        ),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            linecolor=BORDER_COLOR,
            tickfont=dict(size=10, color=SECONDARY_TEXT),
            title_font=dict(size=11, color=SECONDARY_TEXT),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            linecolor=BORDER_COLOR,
            tickfont=dict(size=10, color=SECONDARY_TEXT),
            title_font=dict(size=11, color=SECONDARY_TEXT),
            zeroline=False,
        ),
    )
    return fig


def create_forecast_figure(daily_df: pd.DataFrame, base_dt: pd.Timestamp, forecast_df: pd.DataFrame) -> go.Figure:
    """
    Creates the main interactive forecast figure:
    - 14 days of past actual observations
    - 5 days of multi-horizon predictions with typical error bands (+/- MAE)
    - Subsequent actuals if available
    - CPCB AQI threshold levels
    """
    fig = go.Figure()

    # Past 14 days history
    start_hist = base_dt - timedelta(days=14)
    hist_mask = (daily_df["date"] >= start_hist) & (daily_df["date"] <= base_dt)
    hist_df = daily_df[hist_mask].copy().sort_values("date").reset_index(drop=True)

    # 1. Historical Actual AQI Bars/Line
    hist_x = [d.strftime("%b %d") for d in hist_df["date"]]
    hist_hover = [
        f"<b>{d.strftime('%Y-%m-%d')}</b><br>Observed AQI: {a:.1f}<br>Category: {c}"
        for d, a, c in zip(hist_df["date"], hist_df["aqi"], hist_df["aqi_category"])
    ]

    fig.add_trace(go.Bar(
        x=hist_x,
        y=hist_df["aqi"],
        name="Historical Observed AQI",
        marker=dict(
            color="#A8BDB0",
            line=dict(color="#6C7770", width=0.8),
        ),
        hovertext=hist_hover,
        hoverinfo="text",
    ))

    # 2. Predicted Forecast Bars
    fc_x = [f"{pd.to_datetime(r['target_date']).strftime('%b %d')}<br>(+{r['horizon_num']}d)" for _, r in forecast_df.iterrows()]
    fc_colors = [CATEGORY_COLORS.get(r["category"], PRIMARY_ACCENT) for _, r in forecast_df.iterrows()]
    fc_hover = [
        f"<b>Target: {r['target_date']}</b> ({r['horizon']})<br>"
        f"Predicted AQI: {r['predicted_aqi']:.1f}<br>"
        f"Category: {r['category']}<br>"
        f"Historical Error: ±{r['typical_mae']:.1f} MAE<br>"
        f"Actual Observed: {r['actual_aqi']}"
        for _, r in forecast_df.iterrows()
    ]

    fig.add_trace(go.Bar(
        x=fc_x,
        y=forecast_df["predicted_aqi"],
        name="Random Forest Forecast",
        marker=dict(
            color=fc_colors,
            line=dict(color="#202923", width=1.5),
            pattern=dict(shape="/", fgcolor="#FFFFFF", fillmode="replace", size=6, solidity=0.3)
        ),
        error_y=dict(
            type="data",
            array=forecast_df["typical_mae"],
            visible=True,
            color="#202923",
            thickness=1.2,
            width=5,
        ),
        hovertext=fc_hover,
        hoverinfo="text",
    ))

    # 3. Subsequent Actual observations if present
    subsequent_actuals = forecast_df[forecast_df["has_actual"]]
    if len(subsequent_actuals) > 0:
        sub_x = [f"{pd.to_datetime(r['target_date']).strftime('%b %d')}<br>(+{r['horizon_num']}d)" for _, r in subsequent_actuals.iterrows()]
        fig.add_trace(go.Scatter(
            x=sub_x,
            y=subsequent_actuals["actual_aqi_num"],
            mode="markers+text",
            name="Actual Subsequent AQI",
            marker=dict(size=11, color="#202923", symbol="diamond"),
            text=[f"Act: {val:.0f}" for val in subsequent_actuals["actual_aqi_num"]],
            textposition="top center",
            textfont=dict(size=10, color="#202923"),
            hoverinfo="skip",
        ))

    # 4. Reference lines
    thresholds = [
        (100, "#558B2F", "Satisfactory (100)"),
        (200, "#C2410C", "Poor Alert (200)"),
        (300, "#B91C1C", "Very Poor (300)"),
        (400, "#6B21A8", "Severe (400)"),
    ]
    for val, col, text in thresholds:
        fig.add_hline(
            y=val,
            line_dash="dot",
            line_color=col,
            line_width=1,
            annotation_text=text,
            annotation_position="top left",
            annotation_font=dict(size=9, color=col),
        )

    max_val = max(
        hist_df["aqi"].max() if len(hist_df) else 300,
        forecast_df["predicted_aqi"].max() + 50
    )
    fig.update_yaxes(range=[0, min(530, max_val + 40)], title="Air Quality Index (AQI)")
    fig.update_xaxes(title="Observation / Forecast Target Date")

    base_str = base_dt.strftime("%Y-%m-%d")
    return apply_clean_layout(fig, height=430, title=f"5-Day Multi-Horizon AQI Forecast (Base Date: {base_str})")


def create_historical_trend_chart(filtered_df: pd.DataFrame, pollutant_col: str, title: str, unit: str) -> go.Figure:
    """Historical time-series line chart with range slider."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df[pollutant_col],
        mode="lines",
        name=pollutant_col.upper(),
        line=dict(color=PRIMARY_ACCENT, width=1.5),
        hoverinfo="x+y",
    ))

    # Add 7-day rolling average if dataframe is sufficiently long
    if len(filtered_df) > 14:
        rolling_7 = filtered_df[pollutant_col].rolling(window=7, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=filtered_df["date"],
            y=rolling_7,
            mode="lines",
            name="7-Day Trailing Mean",
            line=dict(color="#C2410C", width=1.8, dash="solid"),
        ))

    # If AQI, add threshold reference lines
    if pollutant_col == "aqi":
        fig.add_hline(y=100, line_dash="dot", line_color="#558B2F", line_width=1, annotation_text="Satisfactory (100)")
        fig.add_hline(y=200, line_dash="dash", line_color="#C2410C", line_width=1.2, annotation_text="Poor Alert (200)")
        fig.add_hline(y=300, line_dash="dot", line_color="#B91C1C", line_width=1, annotation_text="Very Poor (300)")
        fig.add_hline(y=400, line_dash="dash", line_color="#6B21A8", line_width=1.2, annotation_text="Severe (400)")

    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), title="Date")
    fig.update_yaxes(title=f"{title} ({unit})")

    return apply_clean_layout(fig, height=420, title=f"Historical {title} Trend ({unit})")


def create_particulates_comparison_chart(filtered_df: pd.DataFrame) -> go.Figure:
    """Compares PM2.5 and PM10 concentrations over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["pm25"],
        mode="lines",
        name="PM2.5 (Fine Particulate)",
        line=dict(color="#B91C1C", width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["pm10"],
        mode="lines",
        name="PM10 (Coarse Particulate)",
        line=dict(color="#246B57", width=1.4),
    ))

    # CPCB National Ambient Air Quality Standards (NAAQS) 24h limits: PM2.5 = 60, PM10 = 100
    fig.add_hline(y=60, line_dash="dot", line_color="#B91C1C", line_width=1, annotation_text="NAAQS 24h PM2.5 (60 µg/m³)")
    fig.add_hline(y=100, line_dash="dot", line_color="#246B57", line_width=1, annotation_text="NAAQS 24h PM10 (100 µg/m³)")

    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Concentration (µg/m³)")

    return apply_clean_layout(fig, height=380, title="Particulate Matter Comparison: PM2.5 vs PM10")


def create_monthly_seasonal_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Monthly mean AQI bar chart showing strong winter peak."""
    df = daily_df.copy()
    df["month_num"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")

    monthly = df.groupby(["month_num", "month_name"])["aqi"].agg(["mean", "median", "std"]).reset_index()
    monthly = monthly.sort_values("month_num")

    fig = go.Figure()

    colors = [
        "#B91C1C" if m in [11, 12, 1] else "#C2410C" if m in [10, 2] else "#246B57"
        for m in monthly["month_num"]
    ]

    fig.add_trace(go.Bar(
        x=monthly["month_name"],
        y=monthly["mean"],
        error_y=dict(type="data", array=monthly["std"] / 2.0, visible=True, color="#6C7770", thickness=1),
        marker=dict(color=colors, line=dict(color="#202923", width=0.8)),
        name="Mean AQI",
        text=[f"{v:.0f}" for v in monthly["mean"]],
        textposition="outside",
    ))

    fig.add_hline(y=200, line_dash="dash", line_color="#C2410C", annotation_text="Poor Alert Threshold (200)")

    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Mean Daily AQI", range=[0, 360])

    return apply_clean_layout(fig, height=360, title="Monthly Seasonal AQI Distribution (2018–2026)")


def create_category_distribution_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing distribution of AQI categories."""
    cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    counts = daily_df["aqi_category"].value_counts().reindex(cats, fill_value=0)
    pcts = (counts / len(daily_df) * 100.0).round(1)

    fig = go.Figure()

    bar_colors = [CATEGORY_COLORS.get(c, PRIMARY_ACCENT) for c in cats]

    fig.add_trace(go.Bar(
        y=cats,
        x=counts.values,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="#202923", width=0.8)),
        text=[f"{n} days ({p}%)" for n, p in zip(counts.values, pcts.values)],
        textposition="outside",
    ))

    fig.update_xaxes(title="Number of Days")
    fig.update_yaxes(title="CPCB Category", autorange="reversed")

    return apply_clean_layout(fig, height=330, title="AQI Category Distribution Across Complete Dataset")


def create_correlation_heatmap(daily_df: pd.DataFrame) -> go.Figure:
    """Correlation matrix between pollutants and weather variables."""
    cols = ["aqi", "pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed", "barometric_pressure"]
    avail_cols = [c for c in cols if c in daily_df.columns]

    corr = daily_df[avail_cols].corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0.0, "#246B57"],
            [0.5, "#FFFFFF"],
            [1.0, "#B91C1C"],
        ],
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=9, color=PRIMARY_TEXT),
        hoverongaps=False,
    ))

    return apply_clean_layout(fig, height=440, title="Feature Correlation Heatmap (Pollutants & Meteorology)")


def create_evaluation_comparison_bar(metrics_df: pd.DataFrame, metric: str = "MAE") -> go.Figure:
    """Compares Random Forest against all baselines across horizons 1 to 5."""
    fig = go.Figure()

    models_order = ["Random Forest", "Ridge", "Damped Anomaly Persistence", "Climatology", "Persistence"]
    model_colors = {
        "Random Forest": "#246B57",
        "Ridge": "#4E7B24",
        "Damped Anomaly Persistence": "#A16207",
        "Climatology": "#C2410C",
        "Persistence": "#6C7770",
    }

    horizons = [1, 2, 3, 4, 5]
    h_labels = [f"Day +{h}" for h in horizons]

    for model_name in models_order:
        sub = metrics_df[metrics_df["model"] == model_name].sort_values("horizon")
        if len(sub) == 0:
            continue
        vals = [float(sub[sub["horizon"] == h][metric].values[0]) for h in horizons]

        fig.add_trace(go.Bar(
            name=model_name,
            x=h_labels,
            y=vals,
            marker_color=model_colors.get(model_name, "#999999"),
        ))

    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Forecast Horizon")
    fig.update_yaxes(title=f"{metric} (Lower is better)" if metric in ["MAE", "RMSE"] else f"{metric} (Higher is better)")

    return apply_clean_layout(fig, height=390, title=f"Model Comparison Across Forecast Horizons ({metric})")


def create_actual_vs_predicted_scatter(test_preds: pd.DataFrame, horizon: int = 1) -> go.Figure:
    """Scatter plot of actual versus predicted AQI on test set."""
    sub = test_preds[test_preds["horizon"] == horizon].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sub["truth"],
        y=sub["pred_random_forest"],
        mode="markers",
        name="Predictions",
        marker=dict(
            color=PRIMARY_ACCENT,
            size=5,
            opacity=0.6,
        ),
        hovertext=[
            f"Date: {d.strftime('%Y-%m-%d')}<br>Actual: {t:.1f}<br>Predicted: {p:.1f}"
            for d, t, p in zip(sub["date"], sub["truth"], sub["pred_random_forest"])
        ],
        hoverinfo="text",
    ))

    # 1:1 Reference Line
    fig.add_trace(go.Scatter(
        x=[0, 500],
        y=[0, 500],
        mode="lines",
        name="Ideal 1:1 Identity",
        line=dict(color="#B91C1C", width=1.5, dash="dash"),
    ))

    fig.update_xaxes(title="Actual AQI (Truth)", range=[0, 510])
    fig.update_yaxes(title="Predicted AQI (Random Forest)", range=[0, 510])

    return apply_clean_layout(fig, height=400, title=f"Actual vs. Predicted AQI on Test Set (Horizon h={horizon})")


def create_residuals_distribution_chart(test_preds: pd.DataFrame, horizon: int = 1) -> go.Figure:
    """Histogram of test residuals (truth - pred)."""
    sub = test_preds[test_preds["horizon"] == horizon].copy()
    residuals = sub["truth"] - sub["pred_random_forest"]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=40,
        name="Residuals",
        marker=dict(color=PRIMARY_ACCENT, line=dict(color="#FFFFFF", width=0.5)),
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="#B91C1C", line_width=1.5, annotation_text="Zero Error")

    fig.update_xaxes(title="Residual Error (Actual - Predicted AQI Points)")
    fig.update_yaxes(title="Count of Test Days")

    return apply_clean_layout(fig, height=350, title=f"Prediction Residuals Distribution (Horizon h={horizon})")


def create_ci_forest_plot(ci_df: pd.DataFrame, title: str) -> go.Figure:
    """Forest plot displaying difference in MAE with 95% moving-block bootstrap confidence intervals."""
    fig = go.Figure()

    y_labels = [f"Horizon {h}" for h in ci_df["horizon"]]
    diffs = ci_df["mean_diff_mae"]
    lower_err = diffs - ci_df["ci_lower_95"]
    upper_err = ci_df["ci_upper_95"] - diffs

    point_colors = [
        "#1B8755" if hi < 0 else "#6C7770" if lo <= 0 <= hi else "#B91C1C"
        for lo, hi in zip(ci_df["ci_lower_95"], ci_df["ci_upper_95"])
    ]

    fig.add_trace(go.Scatter(
        x=diffs,
        y=y_labels,
        mode="markers",
        error_x=dict(
            type="data",
            symmetric=False,
            array=upper_err,
            arrayminus=lower_err,
            color="#202923",
            thickness=1.5,
            width=6,
        ),
        marker=dict(size=9, color=point_colors),
        name="Mean Difference in MAE",
        hoverinfo="x+y",
    ))

    # Vertical zero line (equivalence boundary)
    fig.add_vline(x=0, line_dash="dash", line_color="#C2410C", line_width=1.5, annotation_text="Zero Difference")

    fig.update_xaxes(title="Δ MAE (RF Error - Baseline Error). Negative = Random Forest is better.")
    fig.update_yaxes(autorange="reversed")

    return apply_clean_layout(fig, height=300, title=title)


def create_shap_family_bar(df_families: pd.DataFrame, mode: str = "absolute") -> go.Figure:
    """Plots stacked bar chart of SHAP feature family contributions across horizons."""
    fig = go.Figure()

    horizons = sorted(df_families["horizon"].unique())
    h_labels = [f"Day +{h}" for h in horizons]
    families = ["Pollutants", "AQI history", "Calendar", "Weather"]

    val_col = "mean_abs_shap" if mode == "absolute" else "share_pct"

    for fam in families:
        sub = df_families[df_families["family"] == fam].set_index("horizon")
        vals = [float(sub.loc[h, val_col]) if h in sub.index else 0.0 for h in horizons]

        fig.add_trace(go.Bar(
            name=fam,
            x=h_labels,
            y=vals,
            marker_color=FAMILY_COLORS.get(fam, "#999999"),
        ))

    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Forecast Horizon")
    y_title = "Sum of Mean |SHAP| (AQI Points)" if mode == "absolute" else "Share of Total Attribution (%)"
    fig.update_yaxes(title=y_title, range=[0, 100] if mode != "absolute" else None)

    return apply_clean_layout(fig, height=380, title=f"SHAP Feature-Family Importance Across Horizons ({mode.capitalize()})")
