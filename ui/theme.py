"""
Design tokens for the AEGISCARE AI interface.

Every colour, size and radius used by ui/styles.py and ui/components.py is
defined here once. .streamlit/config.toml repeats the main colours for
Streamlit's own widgets, so keep the two files in sync.

Colour rules:
- Blue is the only accent (links, active navigation, primary buttons).
- Green / yellow / red are used ONLY for risk or range states, and always
  together with a text label, never as colour alone.
"""

BRAND_NAME = "AEGISCARE"
BRAND_SUFFIX = "AI"
BRAND_TAGLINE = "Integrated Healthcare Safety Intelligence"

COLORS = {
    # Neutrals
    "page": "#F5F7FA",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "border": "#E3E8EF",
    "border_strong": "#CBD5E1",
    "ink": "#0B1F3A",        # deep navy: headings and primary text
    "text": "#24324A",       # body text
    "muted": "#5B6B7F",      # captions, labels
    "subtle": "#8A97A8",     # placeholders, timestamps
    # Accent
    "accent": "#2459C9",
    "accent_hover": "#1C4AAB",
    "accent_soft": "#EAF1FD",
    "accent_border": "#C7D8F6",
}

# Tone -> (text colour, background, border). "success"/"warning"/"danger"
# are the risk colours; "info" uses the accent; "neutral" is grey.
TONES = {
    "neutral": ("#3D4B5F", "#F1F4F8", "#DDE3EA"),
    "info": ("#1C4AAB", "#EAF1FD", "#C7D8F6"),
    "success": ("#14692F", "#E9F6EE", "#BFE3CB"),
    "warning": ("#8A5A0B", "#FEF5E1", "#F2D9A6"),
    "danger": ("#A12626", "#FDEDED", "#F3C4C4"),
}

# Risk levels used by risk_indicator(): level -> (tone, default label, steps lit of 3).
RISK_LEVELS = {
    "low": ("success", "Low", 1),
    "moderate": ("warning", "Moderate", 2),
    "high": ("danger", "High", 3),
    "unknown": ("neutral", "Not assessed", 0),
}

FONT_SANS = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
FONT_MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace"

RADIUS = {"sm": "6px", "md": "10px", "lg": "12px", "pill": "999px"}
SPACE = {"xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "32px"}
SHADOW = {
    "card": "0 1px 2px rgba(11, 31, 58, 0.04), 0 1px 3px rgba(11, 31, 58, 0.06)",
    "raised": "0 4px 12px rgba(11, 31, 58, 0.08)",
}

# Restrained chart palette for Plotly (first colour = primary series).
CHART_COLORS = ["#2459C9", "#0F8B8D", "#7A5AC7", "#C2662D", "#5B6B7F"]
CHART_REFERENCE_BAND = "rgba(36, 89, 201, 0.07)"
CHART_GRID = "#E9EDF2"
