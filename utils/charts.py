"""
Plotly chart helpers for vital-sign trends.

Design rules followed here:
- One y-axis per chart (no dual axes). Each vital sign gets its own chart.
- The shaded band shows the typical adult reference range for context only.
- Colours are fixed per series so they never change between patients.
"""

import plotly.graph_objects as go

from data.synthetic_patients import VITAL_SIGNS
from ui.theme import CHART_COLORS, CHART_GRID, CHART_REFERENCE_BAND, COLORS, FONT_SANS

# Fixed series colours from the ISS palette (blue / teal pair, colour-blind safe).
SERIES_COLORS = {
    "primary": CHART_COLORS[0],
    "secondary": CHART_COLORS[1],
}
REFERENCE_BAND_COLOR = CHART_REFERENCE_BAND

# ISS look shared by every chart: Inter font, transparent background, light grid.
_THEME_LAYOUT = dict(
    font=dict(family=FONT_SANS, size=12, color=COLORS["muted"]),
    title_font=dict(family=FONT_SANS, size=14, color=COLORS["ink"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor=COLORS["surface"], bordercolor=COLORS["border"], font=dict(color=COLORS["ink"])),
)


def _base_layout(fig, title, unit):
    """Shared styling for every trend chart."""
    fig.update_layout(
        **_THEME_LAYOUT,
        title=dict(text=title),
        height=300,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
        yaxis=dict(title=unit, gridcolor=CHART_GRID, zeroline=False),
        xaxis=dict(showgrid=False, linecolor=COLORS["border"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


def _add_reference_band(fig, low, high):
    """Shade the typical reference range behind the data."""
    fig.add_hrect(y0=low, y1=high, fillcolor=REFERENCE_BAND_COLOR, line_width=0, layer="below")


def vital_trend_chart(vitals_df, vital_key):
    """Line chart for a single vital sign with its reference range shaded."""
    info = VITAL_SIGNS[vital_key]
    fig = go.Figure()
    _add_reference_band(fig, info["low"], info["high"])
    fig.add_trace(
        go.Scatter(
            x=vitals_df["timestamp"],
            y=vitals_df[vital_key],
            mode="lines+markers",
            name=info["label"],
            line=dict(color=SERIES_COLORS["primary"], width=2),
            marker=dict(size=6),
            hovertemplate=f"%{{y}} {info['unit']}<extra></extra>",
        )
    )
    return _base_layout(fig, info["label"], info["unit"])


def blood_pressure_chart(vitals_df):
    """Systolic and diastolic pressure on one chart (same unit, same axis)."""
    fig = go.Figure()
    series = [
        ("systolic_bp", "Systolic", SERIES_COLORS["primary"]),
        ("diastolic_bp", "Diastolic", SERIES_COLORS["secondary"]),
    ]
    for key, name, color in series:
        info = VITAL_SIGNS[key]
        _add_reference_band(fig, info["low"], info["high"])
        fig.add_trace(
            go.Scatter(
                x=vitals_df["timestamp"],
                y=vitals_df[key],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"{name}: %{{y}} mmHg<extra></extra>",
            )
        )
        # Direct label at the last point so identity is not colour-only.
        fig.add_annotation(
            x=vitals_df["timestamp"].iloc[-1],
            y=vitals_df[key].iloc[-1],
            text=name,
            showarrow=False,
            xanchor="left",
            xshift=8,
            font=dict(size=11),
        )
    return _base_layout(fig, "Blood pressure", "mmHg")


# ---------------------------------------------------------------------------
# Deterioration Prediction page (simulated demonstration mode)
# ---------------------------------------------------------------------------

def lab_range_chart(labs):
    """
    Where each lab result sits relative to its own reference range.

    `labs` is a dict test -> (value, unit, low, high). Different tests have
    different units, so each value is rescaled: 0 = bottom of the range,
    1 = top of the range (grey band). The real value is shown on hover.
    """
    tests = list(labs)
    positions = [(v - lo) / (hi - lo) for v, _, lo, hi in labs.values()]
    hover = [f"{t}: {v:g} {u} (range {lo:g}–{hi:g})" for t, (v, u, lo, hi) in labs.items()]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=1, fillcolor=REFERENCE_BAND_COLOR, line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=tests,
            mode="markers",
            marker=dict(size=11, color=SERIES_COLORS["primary"]),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        **_THEME_LAYOUT,
        title=dict(text="Lab results relative to their reference range"),
        height=60 + 34 * max(len(tests), 1),
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(
            title="0 = bottom of range · 1 = top of range",
            gridcolor=CHART_GRID,
            zeroline=False,
        ),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def contributions_chart(contributions):
    """Horizontal bars: points each factor added to the demo score."""
    shown = [c for c in contributions if c.points > 0]
    fig = go.Figure(
        go.Bar(
            x=[c.points for c in shown],
            y=[c.label for c in shown],
            orientation="h",
            marker=dict(color=SERIES_COLORS["primary"]),
            text=[f"{c.points:g}" for c in shown],
            textposition="outside",
            customdata=[[c.deviation, c.worsening] for c in shown],
            hovertemplate="%{y}: %{x} points<br>outside range %{customdata[0]}"
            " · worsening %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        **_THEME_LAYOUT,
        title=dict(text="Factors contributing to the demo score"),
        height=80 + 30 * max(len(shown), 1),
        margin=dict(l=10, r=30, t=45, b=10),
        xaxis=dict(title="points", gridcolor=CHART_GRID, zeroline=False),
        yaxis=dict(autorange="reversed"),
        bargap=0.35,
    )
    return fig


def contribution_breakdown_chart(contributions):
    """Horizontal stacked bars: each factor's points split into outside-range and worsening."""
    shown = [c for c in contributions if c.points > 0]
    labels = [c.label for c in shown]
    fig = go.Figure()
    for name, attr, color in (
        ("Outside range", "deviation", SERIES_COLORS["primary"]),
        ("Worsening (24 h)", "worsening", SERIES_COLORS["secondary"]),
    ):
        fig.add_trace(go.Bar(
            x=[getattr(c, attr) for c in shown], y=labels, orientation="h", name=name,
            marker=dict(color=color), hovertemplate=f"%{{y}}: %{{x}} points ({name.lower()})<extra></extra>",
        ))
    # Total points at the end of each bar.
    fig.add_trace(go.Scatter(
        x=[c.points for c in shown], y=labels, mode="text", text=[f"  {c.points:g}" for c in shown],
        textposition="middle right", showlegend=False, hoverinfo="skip",
        textfont=dict(color=COLORS["ink"], size=12),
    ))
    fig.update_layout(
        **_THEME_LAYOUT,
        barmode="stack",
        height=90 + 34 * max(len(shown), 1),
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(title="points (relative contribution)", gridcolor=CHART_GRID, zeroline=False,
                   range=[0, max([c.points for c in shown] or [1]) * 1.2]),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, traceorder="normal"),
        bargap=0.4,
    )
    return fig


def score_history_chart(history):
    """Line chart of the vital-sign part of the demo score over time."""
    fig = go.Figure(
        go.Scatter(
            x=history["timestamp"],
            y=history["score"],
            mode="lines+markers",
            line=dict(color=SERIES_COLORS["primary"], width=2),
            marker=dict(size=6),
            hovertemplate="%{y} points<extra></extra>",
            showlegend=False,
        )
    )
    return _base_layout(fig, "Vital-sign part of the demo score over time", "points")
