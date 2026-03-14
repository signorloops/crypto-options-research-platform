"""Plotly chart building functions for all dashboard pages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Overview charts ──────────────────────────────────────────────────

def timeseries_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> str:
    """Build a Plotly line chart and return embedded HTML (no full page)."""
    fig = px.line(df, x=x, y=y, title=title, template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def returns_histogram(series: pd.Series, title: str) -> str:
    """Build a returns distribution histogram."""
    fig = px.histogram(series, nbins=40, title=title, template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def deviation_heatmap(heatmap_df: pd.DataFrame) -> str:
    """Build deviation heatmap HTML."""
    fig = px.density_heatmap(
        heatmap_df,
        x="delta_bucket",
        y="expiry_bucket",
        z="abs_deviation_bps",
        histfunc="avg",
        color_continuous_scale="RdBu_r",
        title="Cross-Market Deviation Heatmap (abs bps)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Backtest charts ──────────────────────────────────────────────────

def pnl_line_chart(pnl_data: List[List[Any]], title: str) -> str:
    """Cumulative PnL line chart from [[timestamp, value], ...] data."""
    if not pnl_data:
        return "<p>No PnL data available.</p>"
    timestamps = [row[0] for row in pnl_data]
    values = [row[1] for row in pnl_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=values, mode="lines", name="Cumulative PnL"))
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Time", yaxis_title="PnL")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def position_chart(position_data: List[List[Any]], title: str) -> str:
    """Position (inventory) line chart with zero line."""
    if not position_data:
        return "<p>No position data available.</p>"
    timestamps = [row[0] for row in position_data]
    values = [row[1] for row in position_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=values, mode="lines", name="Position", fill="tozeroy"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Time", yaxis_title="Position")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def pnl_sampled_chart(pnl_values: List[float], title: str) -> str:
    """Sampled PnL line chart from [val, val, ...] data (full_history format)."""
    if not pnl_values:
        return "<p>No PnL data available.</p>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pnl_values, mode="lines", name="Cumulative PnL"))
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Sample", yaxis_title="PnL")
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Strategy comparison charts ───────────────────────────────────────

def multi_pnl_overlay(strategies_data: Dict[str, Dict[str, Any]]) -> str:
    """Overlay PnL curves for multiple strategies."""
    fig = go.Figure()
    for name, data in strategies_data.items():
        pnl = data.get("pnl_history", [])
        sampled = data.get("pnl_history_sampled", [])
        if pnl:
            timestamps = [row[0] for row in pnl]
            values = [row[1] for row in pnl]
            fig.add_trace(go.Scatter(x=timestamps, y=values, mode="lines", name=name))
        elif sampled:
            fig.add_trace(go.Scatter(y=sampled, mode="lines", name=name))
    fig.update_layout(
        title="Cumulative PnL Comparison",
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="PnL",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def comparison_bars(
    names: List[str],
    values: List[float],
    title: str,
    y_label: str = "Value",
) -> str:
    """Generic bar chart comparing a single metric across strategies."""
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=values, marker_color=bar_colors))
    fig.update_layout(title=title, template="plotly_white", yaxis_title=y_label)
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Risk charts ──────────────────────────────────────────────────────

def circuit_breaker_badge(state: str) -> str:
    """Render an HTML badge for circuit breaker state."""
    state_upper = state.upper()
    css_class = {
        "NORMAL": "badge-normal",
        "WARNING": "badge-warning",
        "RESTRICTED": "badge-restricted",
        "HALTED": "badge-halted",
    }.get(state_upper, "badge-warning")
    return f'<span class="badge {css_class}">{state_upper}</span>'


def regime_badge(regime: str) -> str:
    """Render an HTML badge for regime state."""
    regime_upper = regime.upper()
    css_class = {
        "LOW": "badge-low",
        "MEDIUM": "badge-medium",
        "HIGH": "badge-high",
    }.get(regime_upper, "badge-medium")
    return f'<span class="badge {css_class}">{regime_upper}</span>'


def greeks_bar_chart(greeks: Dict[str, float]) -> str:
    """Bar chart of Greeks exposures."""
    if not greeks:
        return "<p>No Greeks data available.</p>"
    names = list(greeks.keys())
    values = list(greeks.values())
    colors = ["#3b82f6" if v >= 0 else "#ef4444" for v in values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=values, marker_color=colors))
    fig.update_layout(
        title="Greeks Exposure",
        template="plotly_white",
        yaxis_title="Exposure",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)
