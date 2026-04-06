"""
EARTHLENS AI — Plots Module
=============================
Author : Gouragopal Mohapatra
Purpose: Generate charts and graphs from analysis results
Library: Plotly
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger


# ── NDVI Histogram ─────────────────────────────────────────────────────────────
def ndvi_histogram(ndvi: np.ndarray) -> go.Figure:
    """
    Plot NDVI value distribution histogram.
    """
    valid = ndvi[~np.isnan(ndvi)].flatten()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x          = valid,
        nbinsx     = 50,
        name       = "NDVI Distribution",
        marker     = dict(
            color = valid,
            colorscale = [
                [0.0,  "#2166ac"],   # Water
                [0.25, "#d4a56a"],   # Bare
                [0.5,  "#d9ef8b"],   # Sparse
                [0.75, "#66bd63"],   # Moderate
                [1.0,  "#1a7837"],   # Dense
            ],
            line = dict(width=0),
        ),
        opacity = 0.85,
    ))

    fig.update_layout(
        title      = "NDVI Value Distribution",
        xaxis_title= "NDVI Value",
        yaxis_title= "Pixel Count",
        template   = "plotly_dark",
        bargap     = 0.05,
        showlegend = False,
        height     = 350,
        margin     = dict(l=40, r=20, t=50, b=40),
    )

    # Add class boundary lines
    boundaries = [0.0, 0.2, 0.4, 0.6]
    labels     = ["Water|Bare", "Bare|Sparse", "Sparse|Mod", "Mod|Dense"]

    for b, label in zip(boundaries, labels):
        fig.add_vline(
            x            = b,
            line_dash    = "dash",
            line_color   = "rgba(255,255,255,0.4)",
            annotation_text     = label,
            annotation_position = "top",
            annotation_font_size= 9,
        )

    logger.info("NDVI histogram generated")
    return fig


# ── NDVI Coverage Pie Chart ────────────────────────────────────────────────────
def ndvi_coverage_pie(stats: dict) -> go.Figure:
    """
    Pie chart of NDVI land cover class coverage.
    Input: stats dict from ndvi.ndvi_stats()
    """
    coverage = stats.get("coverage", {})

    labels = list(coverage.keys())
    values = list(coverage.values())
    colors = [
        "#2166ac",   # Water
        "#d4a56a",   # Bare/Urban
        "#d9ef8b",   # Sparse
        "#66bd63",   # Moderate
        "#1a7837",   # Dense
    ]

    fig = go.Figure(go.Pie(
        labels           = labels,
        values           = values,
        marker_colors    = colors,
        hole             = 0.4,
        textinfo         = "label+percent",
        textfont_size    = 11,
        hovertemplate    = "<b>%{label}</b><br>Coverage: %{value:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title    = "Land Cover Distribution (NDVI)",
        template = "plotly_dark",
        height   = 350,
        margin   = dict(l=20, r=20, t=50, b=20),
        showlegend = False,
    )

    logger.info("NDVI coverage pie chart generated")
    return fig


# ── NDVI Stats Bar Chart ───────────────────────────────────────────────────────
def ndvi_stats_bar(stats: dict) -> go.Figure:
    """
    Bar chart showing NDVI statistics.
    """
    metrics = {
        "Min"  : stats.get("min",  0),
        "Mean" : stats.get("mean", 0),
        "Max"  : stats.get("max",  0),
        "Std"  : stats.get("std",  0),
    }

    colors = ["#e41a1c", "#ff7f00", "#1a7837", "#984ea3"]

    fig = go.Figure(go.Bar(
        x      = list(metrics.keys()),
        y      = list(metrics.values()),
        marker_color = colors,
        text   = [f"{v:.3f}" for v in metrics.values()],
        textposition = "outside",
    ))

    fig.update_layout(
        title      = "NDVI Statistics",
        yaxis_title= "Value",
        template   = "plotly_dark",
        height     = 300,
        margin     = dict(l=40, r=20, t=50, b=40),
        yaxis      = dict(range=[-1.2, 1.2]),
    )

    logger.info("NDVI stats bar chart generated")
    return fig


# ── Water Coverage Bar ─────────────────────────────────────────────────────────
def water_coverage_bar(stats: dict) -> go.Figure:
    """
    Bar chart showing water vs non-water coverage.
    """
    water_pct    = stats.get("water_coverage", 0)
    no_water_pct = round(100 - water_pct, 2)

    fig = go.Figure(go.Bar(
        x      = ["Water", "No Water"],
        y      = [water_pct, no_water_pct],
        marker_color = ["#2166ac", "#d4a56a"],
        text   = [f"{water_pct}%", f"{no_water_pct}%"],
        textposition = "outside",
    ))

    fig.update_layout(
        title      = "Water Coverage",
        yaxis_title= "Percentage (%)",
        template   = "plotly_dark",
        height     = 300,
        margin     = dict(l=40, r=20, t=50, b=40),
        yaxis      = dict(range=[0, 110]),
    )

    logger.info("Water coverage bar chart generated")
    return fig


# ── Combined NDVI Dashboard ────────────────────────────────────────────────────
def ndvi_dashboard(ndvi: np.ndarray, stats: dict) -> go.Figure:
    """
    Combined dashboard — histogram + pie + stats bar.
    """
    fig = make_subplots(
        rows = 1, cols = 3,
        subplot_titles = (
            "NDVI Distribution",
            "Land Cover %",
            "NDVI Stats",
        ),
        specs = [[
            {"type": "histogram"},
            {"type": "pie"},
            {"type": "bar"},
        ]],
    )

    valid = ndvi[~np.isnan(ndvi)].flatten()

    # Histogram
    fig.add_trace(go.Histogram(
        x      = valid,
        nbinsx = 40,
        marker_color = "#66bd63",
        name   = "Distribution",
    ), row=1, col=1)

    # Pie
    coverage = stats.get("coverage", {})
    colors   = ["#2166ac","#d4a56a","#d9ef8b","#66bd63","#1a7837"]
    fig.add_trace(go.Pie(
        labels        = list(coverage.keys()),
        values        = list(coverage.values()),
        marker_colors = colors,
        hole          = 0.35,
        textinfo      = "percent",
        showlegend    = False,
    ), row=1, col=2)

    # Stats bar
    metrics = {
        "Min" : stats.get("min",  0),
        "Mean": stats.get("mean", 0),
        "Max" : stats.get("max",  0),
    }
    fig.add_trace(go.Bar(
        x    = list(metrics.keys()),
        y    = list(metrics.values()),
        marker_color = ["#e41a1c", "#ff7f00", "#1a7837"],
        text = [f"{v:.3f}" for v in metrics.values()],
        textposition = "outside",
        showlegend   = False,
    ), row=1, col=3)

    fig.update_layout(
        title    = "🌿 NDVI Analysis Dashboard",
        template = "plotly_dark",
        height   = 400,
        margin   = dict(l=40, r=40, t=80, b=40),
    )

    logger.info("NDVI dashboard generated")
    return fig