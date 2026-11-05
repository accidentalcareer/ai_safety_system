"""
Global CSS for the AEGISCARE AI interface.

inject_css() is called once per run from app.py. It does two things:
1. Restyles Streamlit's native elements (metrics, tabs, alerts, buttons,
   expanders, sidebar navigation) so every page looks consistent.
2. Defines the `.iss-*` classes used by ui/components.py.

All values come from ui/theme.py through CSS custom properties, so the
stylesheet itself contains no hard-coded colours.
"""

import streamlit as st

from ui.theme import COLORS, FONT_MONO, FONT_SANS, RADIUS, SHADOW, SPACE, TONES


def _root_variables():
    """Turn the design tokens into CSS custom properties on :root."""
    lines = [f"--iss-{name.replace('_', '-')}: {value};" for name, value in COLORS.items()]
    for tone, (fg, bg, border) in TONES.items():
        lines += [f"--iss-{tone}-fg: {fg};", f"--iss-{tone}-bg: {bg};", f"--iss-{tone}-border: {border};"]
    lines += [f"--iss-radius-{k}: {v};" for k, v in RADIUS.items()]
    lines += [f"--iss-space-{k}: {v};" for k, v in SPACE.items()]
    lines += [f"--iss-shadow-{k}: {v};" for k, v in SHADOW.items()]
    lines += [f"--iss-font-sans: {FONT_SANS};", f"--iss-font-mono: {FONT_MONO};"]
    return ":root {" + "".join(lines) + "}"


_CSS = """
/* ---------------------------------------------------------------- layout */
[data-testid="stAppViewContainer"] { background: var(--iss-page); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
    max-width: 1240px;
    padding-top: 2.25rem;
    padding-bottom: 4rem;
}

/* ------------------------------------------------------------ typography */
h1, h2, h3, h4 { color: var(--iss-ink); letter-spacing: -0.01em; }
[data-testid="stHeading"] h1 { padding-bottom: 0.15rem; }
/* Captions: the muted token at full opacity (Streamlit fades them to 0.6). */
[data-testid="stCaptionContainer"] { color: var(--iss-muted); opacity: 1; }
code { font-family: var(--iss-font-mono); }
/* Streamlit pulls markdown blocks up by 1rem to offset paragraph margins.
   Component HTML has no paragraph, so cancel that to keep spacing even. */
[data-testid="stMarkdownContainer"]:has([class^="iss-"]) { margin-bottom: 0; }
hr { border-color: var(--iss-border) !important; margin: 1.5rem 0 !important; }

/* ---------------------------------------------------------------- cards */
/* One surface for every top-level card: st.metric, keyed card containers
   (ui.components.card_container gives them a key starting with "iss_card_",
   exposed as the class "st-key-iss_card_..."), HTML cards and metric cards. */
[data-testid="stMetric"], [class*="st-key-iss_card_"], .iss-card, .iss-metric {
    background: var(--iss-surface);
    border: 1px solid var(--iss-border);
    border-radius: var(--iss-radius-lg);
    box-shadow: var(--iss-shadow-card);
}
[class*="st-key-iss_card_"], .iss-card { padding: 18px 20px; }
/* Supporting information (card_container(muted=True)): tinted, flat. */
[class*="st-key-iss_card_"][class*="__muted"] { background: var(--iss-surface-alt); box-shadow: none; }
/* Anything card-like inside a card is flat, so depth never stacks up. */
[class*="st-key-iss_card_"] [class*="st-key-iss_card_"],
[class*="st-key-iss_card_"] .iss-metric,
[class*="st-key-iss_card_"] [data-testid="stMetric"] { box-shadow: none; }
[class*="st-key-iss_card_"] h3 { font-size: 1.08rem; font-weight: 600; padding: 0 0 0.4rem; }
[class*="st-key-iss_card_"] h4 { font-size: 1.05rem; font-weight: 600; padding: 0.35rem 0 0.4rem; }

/* --------------------------------------------------------------- metrics */
[data-testid="stMetricLabel"] { color: var(--iss-muted); font-weight: 500; }
[data-testid="stMetricValue"] { color: var(--iss-ink); font-variant-numeric: tabular-nums; }

/* ------------------------------------------------------------------ tabs */
[data-testid="stTabs"] [role="tablist"] { gap: 4px; border-bottom: 1px solid var(--iss-border); }
[data-testid="stTab"] { padding: 8px 14px; color: var(--iss-muted); font-weight: 500; }
[data-testid="stTab"][aria-selected="true"] { color: var(--iss-accent); }
[data-testid="stTab"]:hover { color: var(--iss-ink); }

/* ---------------------------------------------------------------- alerts */
/* Streamlit's alert kinds use the same tone tokens as badges and panels;
   body text stays dark for readability and the icon carries the tone colour. */
[data-testid="stAlertContainer"] { border-radius: var(--iss-radius-md); border: 1px solid var(--iss-border); }
[data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] { color: var(--iss-text); }
[data-testid="stAlertContainer"] strong { color: var(--iss-ink); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background: var(--iss-info-bg); border-color: var(--iss-info-border); }
/* Success = an action completed, not "safe": calm info palette, not risk green. */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { background: var(--iss-info-bg); border-color: var(--iss-info-border); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { background: var(--iss-warning-bg); border-color: var(--iss-warning-border); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { background: var(--iss-danger-bg); border-color: var(--iss-danger-border); }
[data-testid="stAlertContentInfo"] [data-testid="stAlertDynamicIcon"] { color: var(--iss-info-fg); }
[data-testid="stAlertContentSuccess"] [data-testid="stAlertDynamicIcon"] { color: var(--iss-info-fg); }
[data-testid="stAlertContentWarning"] [data-testid="stAlertDynamicIcon"] { color: var(--iss-warning-fg); }
[data-testid="stAlertContentError"] [data-testid="stAlertDynamicIcon"] { color: var(--iss-danger-fg); }
/* Page-level demo notices (key "iss_notice_..."): one style on every page,
   whatever alert kind the page uses (Deterioration keeps st.error). Being a
   demo is not a risk, so they use slate with an accent bar, not a risk colour. */
[class*="st-key-iss_notice"] [data-testid="stAlertContainer"] {
    background: var(--iss-neutral-bg); border-color: var(--iss-neutral-border);
    box-shadow: inset 3px 0 0 var(--iss-accent);
}
[class*="st-key-iss_notice"] [data-testid="stAlertDynamicIcon"] { color: var(--iss-accent); }

/* --------------------------------------------------------------- buttons */
[data-testid^="stBaseButton-"] { font-weight: 500; transition: background 0.15s, border-color 0.15s; }
[data-testid="stBaseButton-secondary"] { background: var(--iss-surface); border-color: var(--iss-border-strong); }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--iss-accent); color: var(--iss-accent); }

/* ---------------------------------------------------------------- inputs */
/* Multiselect chips: soft accent like the badges, not solid primary blue. */
[data-testid="stMultiSelectTagsContainer"] [data-tag] {
    background: var(--iss-accent-soft);
    border: 1px solid var(--iss-accent-border);
    color: var(--iss-accent);
    font-weight: 500;
}
[data-testid="stMultiSelectTagsContainer"] [data-tag] * { color: var(--iss-accent); }

/* ------------------------------------------------------------- expanders */
[data-testid="stExpander"] details {
    background: var(--iss-surface);
    border: 1px solid var(--iss-border);
    border-radius: var(--iss-radius-md);
}
[data-testid="stExpander"] summary { font-weight: 500; color: var(--iss-ink); }

/* --------------------------------------------------------------- sidebar */
[data-testid="stSidebarHeader"] { height: 2.25rem; padding-bottom: 0; }
[data-testid="stSidebarUserContent"] { padding-top: 0; padding-bottom: 1.5rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.6rem; }
[data-testid="stSidebar"] hr { margin: 0.6rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 0.8rem; color: var(--iss-muted); }

/* The page radio (key "page") stays the single source of routing state but
   is not shown; the grouped nav buttons below set its value. */
.st-key-iss_nav_state { display: none !important; }

/* Grouped navigation: compact full-width rows, left-aligned. */
.st-key-iss_nav { gap: 2px !important; }
.st-key-iss_nav .iss-label { margin: 10px 0 2px 10px; }
.st-key-iss_nav > div:first-child .iss-label { margin-top: 0; }
[data-testid="stSidebar"] .iss-label { margin-left: 10px; }
[class*="st-key-nav_"] button {
    width: 100%;
    min-height: 34px;
    padding: 6px 10px;
    justify-content: flex-start;
    border: 1px solid transparent;
    border-radius: var(--iss-radius-md);
    background: transparent;
    color: var(--iss-text);
    box-shadow: none;
}
[class*="st-key-nav_"] button > div { justify-content: flex-start; gap: 10px; }
[class*="st-key-nav_"] button p { font-size: 0.9rem; font-weight: 500; }
[class*="st-key-nav_"] button span[data-testid="stIconMaterial"] { color: var(--iss-subtle); font-size: 1.1rem; }
[class*="st-key-nav_"] button:hover { background: var(--iss-surface-alt); color: var(--iss-ink); border-color: transparent; }
/* Active page: soft blue fill, blue text and a left accent bar. */
[class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] {
    background: var(--iss-accent-soft);
    border-color: var(--iss-accent-border);
    color: var(--iss-accent);
    box-shadow: inset 3px 0 0 var(--iss-accent);
}
[class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] p { font-weight: 600; }
[class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] span[data-testid="stIconMaterial"] { color: var(--iss-accent); }
[class*="st-key-nav_"] button:focus-visible { outline: 2px solid var(--iss-accent); outline-offset: 1px; }

/* =========================================================================
   Component classes (ui/components.py)
   ========================================================================= */

/* Sidebar brand */
.iss-brand { display: flex; align-items: center; gap: 10px; padding: 0 2px 4px; }
.iss-brand-mark {
    width: 34px; height: 34px; flex: none;
    display: flex; align-items: center; justify-content: center;
    border-radius: var(--iss-radius-md);
    background: var(--iss-ink);
}
.iss-brand-name { font-weight: 700; font-size: 1rem; color: var(--iss-ink); line-height: 1.15; letter-spacing: 0.04em; }
.iss-brand-name span { color: var(--iss-accent); }
.iss-brand-tagline { font-size: 0.72rem; color: var(--iss-muted); line-height: 1.3; margin-top: 1px; }

/* Sidebar system status */
.iss-status { display: flex; flex-direction: column; gap: 9px; padding: 0 10px; }
/* Dot + name on the first line, the value underneath: long values wrap cleanly. */
.iss-status-row { display: grid; grid-template-columns: 7px 1fr; column-gap: 8px; align-items: center; font-size: 0.84rem; }
.iss-status-name { color: var(--iss-ink); font-weight: 500; line-height: 1.3; }
.iss-status-value { grid-column: 2; color: var(--iss-muted); font-size: 0.76rem; line-height: 1.35; }
.iss-status-dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.iss-demo {
    margin: 4px 0 0; padding: 10px 12px;
    border: 1px dashed var(--iss-border-strong); border-radius: var(--iss-radius-md);
    background: var(--iss-surface-alt);
}
.iss-demo-title { font-size: 0.8rem; font-weight: 600; color: var(--iss-ink); }
.iss-demo-text { font-size: 0.74rem; color: var(--iss-muted); line-height: 1.4; margin-top: 2px; }

/* Small uppercase labels: one style for every component that uses them. */
.iss-label, .iss-eyebrow, .iss-kv-label, .iss-flow-label, .iss-sgrid-name, .iss-mode-state {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--iss-muted); line-height: 1.4;
}
.iss-label { margin: 6px 0; }

/* Page header: page_header (st.title in a keyed container) and page_title
   (HTML, Dashboard) render the same eyebrow / title / subtitle / badges. */
.iss-eyebrow { color: var(--iss-accent); font-size: 0.72rem; }
.st-key-iss_page_head { gap: 4px; margin-bottom: 4px; }
.st-key-iss_page_head h1 { font-size: 1.6rem; line-height: 1.2; padding: 0; }
.st-key-iss_page_head [data-testid="stHeading"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
.st-key-iss_page_head [data-testid="stCaptionContainer"] p { font-size: 0.9rem; }
.st-key-iss_page_head .iss-badge-row { margin: 6px 0 0; }
.iss-badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 6px; }

.iss-page-head { margin-bottom: 4px; }
.iss-page-head .iss-eyebrow { margin-bottom: 4px; }
.iss-page-title { font-size: 1.6rem; font-weight: 650; color: var(--iss-ink); letter-spacing: -0.01em; line-height: 1.2; }
.iss-page-sub { font-size: 0.9rem; color: var(--iss-muted); margin-top: 4px; }
.iss-page-head .iss-badge-row { margin-top: 10px; }

/* Card containers with a tone edge (card_container(tone=...)) */
[class*="st-key-iss_card_"][class*="__neutral"] { border-top: 3px solid var(--iss-border-strong); }
[class*="st-key-iss_card_"][class*="__info"] { border-top: 3px solid var(--iss-accent); }
[class*="st-key-iss_card_"][class*="__success"] { border-top: 3px solid var(--iss-success-fg); }
[class*="st-key-iss_card_"][class*="__warning"] { border-top: 3px solid var(--iss-warning-fg); }
[class*="st-key-iss_card_"][class*="__danger"] { border-top: 3px solid var(--iss-danger-fg); }

/* Patient Dashboard identity card: the patient name is the page's st.title. */
.st-key-iss_card_identity h1 { font-size: 1.45rem; padding: 0 0 2px; }

/* Equal-height cards in a row (card_container(fill_height=True)). Columns
   already stretch to the tallest one; the card fills its column. */
[data-testid="stColumn"]:has([class*="st-key-iss_card_"][class*="__fill"]) > [data-testid="stVerticalBlock"] { height: 100%; }
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-iss_card_"][class*="__fill"]) { flex: 1; }
[class*="st-key-iss_card_"][class*="__fill"] { flex: 1; }
[class*="st-key-iss_card_"][class*="__fill"] > :last-child:has(button) { margin-top: auto; }

/* Column rows that wrap instead of squeezing (a container keyed "iss_wrap_..."):
   each column keeps a readable minimum width and moves to a new row when it cannot. */
[class*="st-key-iss_wrap"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
[class*="st-key-iss_wrap"] [data-testid="stColumn"] { min-width: 240px; }
/* Badges inside summary cards may wrap, so a long badge never sticks out of its card. */
.iss-summary .iss-badge { white-space: normal; }

/* Alert cards: coloured left edge instead of top (card_container(edge="left")). */
[class*="st-key-iss_card_"][class*="__left"] { border-top: 1px solid var(--iss-border); border-left-width: 4px; }
[class*="st-key-iss_card_"][class*="__warning__left"] { border-left-color: var(--iss-warning-fg); }
[class*="st-key-iss_card_"][class*="__danger__left"] { border-left-color: var(--iss-danger-fg); }
[class*="st-key-iss_card_"][class*="__info__left"] { border-left-color: var(--iss-accent); }
[class*="st-key-iss_card_"][class*="__neutral__left"] { border-left-color: var(--iss-border-strong); }
[class*="st-key-iss_card_"][class*="__success__left"] { border-left-color: var(--iss-success-fg); }

/* Nested cards (inside a card or a grid): one surface, medium radius, no shadow. */
.iss-tile, .iss-flow-step, .iss-sgrid-item {
    background: var(--iss-surface); border: 1px solid var(--iss-border); border-radius: var(--iss-radius-md);
}

/* Tiles (tile_html / tile_grid) */
.iss-tile-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }
.iss-tile {
    border-left: 3px solid var(--iss-border); padding: 12px 14px;
}
.iss-tile-title { font-weight: 600; color: var(--iss-ink); font-size: 0.95rem; }
.iss-tile-sub { font-size: 0.8rem; color: var(--iss-text); margin-top: 2px; }
.iss-tile-detail { font-size: 0.78rem; color: var(--iss-muted); margin-top: 2px; }
.iss-tile .iss-badge-row { margin: 8px 0 0; }

/* Fact rows (fact_rows) */
.iss-facts { border: 1px solid var(--iss-border); border-radius: var(--iss-radius-md); overflow: hidden; }
.iss-fact {
    display: grid; grid-template-columns: minmax(110px, 1fr) minmax(90px, 1fr) 2fr; gap: 12px;
    padding: 7px 12px; font-size: 0.84rem; border-bottom: 1px solid var(--iss-border); background: var(--iss-surface);
}
.iss-fact:last-child { border-bottom: none; }
.iss-fact-item { color: var(--iss-text); font-weight: 500; }
.iss-fact-value { color: var(--iss-ink); font-variant-numeric: tabular-nums; }
.iss-fact-missing { color: var(--iss-subtle); font-style: italic; }
.iss-fact-note { color: var(--iss-muted); }

/* Summary block (summary_block) */
.iss-summary .iss-label { margin: 0 0 6px; }
.iss-summary-headline { font-size: 1.08rem; font-weight: 600; color: var(--iss-ink); line-height: 1.35; }
.iss-summary-detail { font-size: 0.84rem; color: var(--iss-muted); margin-top: 4px; line-height: 1.45; }
.iss-summary .iss-badge-row { margin: 10px 0 0; }
.iss-summary-rows { margin-top: 10px; border-top: 1px solid var(--iss-border); }
.iss-summary-row {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 6px 0; font-size: 0.84rem; border-bottom: 1px solid var(--iss-border);
}
.iss-summary-row span:first-child { color: var(--iss-text); }
.iss-summary-row span:last-child { color: var(--iss-muted); text-align: right; }
.iss-summary-row:last-child { border-bottom: none; }

/* Section header */
.iss-section { display: flex; align-items: flex-start; gap: 12px; margin: 28px 0 12px; }
.iss-section-step {
    flex: none; width: 26px; height: 26px; border-radius: var(--iss-radius-pill);
    background: var(--iss-accent-soft); color: var(--iss-accent); border: 1px solid var(--iss-accent-border);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 600; margin-top: 1px;
}
.iss-section-title { font-size: 1.15rem; font-weight: 600; color: var(--iss-ink); line-height: 1.4; }
.iss-section-desc { font-size: 0.88rem; color: var(--iss-muted); margin-top: 2px; }
.iss-section .iss-badge-row { margin: 6px 0 0; }

/* Card (surface in the shared card rule) */
.iss-card { margin-bottom: 12px; }
.iss-card-accent { border-top: 3px solid var(--iss-accent); }
.iss-card-title { font-weight: 600; color: var(--iss-ink); font-size: 1rem; margin-bottom: 6px; }
.iss-card-body { color: var(--iss-text); font-size: 0.92rem; line-height: 1.55; }
.iss-card-footer { color: var(--iss-muted); font-size: 0.8rem; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--iss-border); }

/* Metric card */
.iss-metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 12px; }
.iss-metric { padding: 14px 16px; border-left-width: 3px; }
.iss-metric-label { font-size: 0.8rem; font-weight: 500; color: var(--iss-muted); }
.iss-metric-value {
    font-size: 1.6rem; font-weight: 600; color: var(--iss-ink); line-height: 1.25; margin-top: 4px;
    font-variant-numeric: tabular-nums;
}
.iss-metric-unit { font-size: 0.85rem; font-weight: 500; color: var(--iss-muted); margin-left: 4px; }
.iss-metric-foot { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-top: 8px; font-size: 0.78rem; color: var(--iss-muted); }

/* Status badge */
.iss-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 2px 10px; border-radius: var(--iss-radius-pill);
    font-size: 0.75rem; font-weight: 600; line-height: 1.6; white-space: nowrap;
    border: 1px solid;
}
.iss-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex: none; }

/* Risk indicator */
.iss-risk { display: flex; align-items: center; gap: 12px; }
.iss-risk-scale { display: flex; gap: 3px; flex: none; }
.iss-risk-step { width: 22px; height: 8px; border-radius: 3px; background: var(--iss-border); }
.iss-risk-text { display: flex; flex-direction: column; }
.iss-risk-label { font-weight: 600; font-size: 0.9rem; }
.iss-risk-detail { font-size: 0.8rem; color: var(--iss-muted); }

/* Information panel */
.iss-panel {
    display: flex; gap: 12px;
    border: 1px solid; border-radius: var(--iss-radius-md);
    padding: 12px 16px; margin-bottom: 12px;
    font-size: 0.9rem; line-height: 1.55;
}
.iss-panel-bar { width: 3px; border-radius: 2px; background: currentColor; flex: none; }
.iss-panel-title { font-weight: 600; margin-bottom: 2px; }
.iss-panel-body { color: var(--iss-text); }

/* Key-value list */
.iss-kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px 24px; }
.iss-kv-value { font-size: 0.95rem; color: var(--iss-ink); margin-top: 2px; }

/* Timeline */
.iss-timeline { position: relative; padding-left: 4px; }
.iss-tl-entry { position: relative; display: flex; gap: 14px; padding-bottom: 18px; }
.iss-tl-entry:not(:last-child)::before {
    content: ""; position: absolute; left: 5px; top: 16px; bottom: 0;
    width: 2px; background: var(--iss-border);
}
.iss-tl-dot {
    flex: none; width: 12px; height: 12px; margin-top: 4px; border-radius: 50%;
    border: 2px solid; background: var(--iss-surface); position: relative; z-index: 1;
}
.iss-tl-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; }
.iss-tl-title { font-weight: 600; color: var(--iss-ink); font-size: 0.92rem; }
.iss-tl-time { font-size: 0.78rem; color: var(--iss-subtle); font-family: var(--iss-font-mono); }
.iss-tl-body { font-size: 0.86rem; color: var(--iss-text); margin-top: 2px; line-height: 1.5; }
.iss-tl-meta { font-size: 0.78rem; color: var(--iss-muted); margin-top: 4px; }

.iss-mono { font-family: var(--iss-font-mono); font-size: 0.92em; }
.iss-muted { color: var(--iss-muted); }

/* Mode strip (mode_strip): which kind of output a result is. */
.iss-mode { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.iss-mode-item {
    border: 1px dashed var(--iss-border-strong); border-radius: var(--iss-radius-md);
    padding: 8px 12px; color: var(--iss-subtle); background: var(--iss-surface);
}
.iss-mode-item.is-active { border-style: solid; border-width: 1px 1px 1px 4px; }
.iss-mode-item .iss-mode-state { color: inherit; }
.iss-mode-title { font-weight: 600; font-size: 0.92rem; margin-top: 2px; }
.iss-mode-item:not(.is-active) .iss-mode-title { color: var(--iss-muted); }
.iss-mode-detail { font-size: 0.78rem; color: var(--iss-muted); margin-top: 2px; line-height: 1.4; }
@media (max-width: 900px) { .iss-mode { grid-template-columns: 1fr; } }

/* Deterioration and ICU Risk pages: large score inside the result card. */
[class*="st-key-iss_card_det_risk"] [data-testid="stMetric"],
[class*="st-key-iss_card_icu_result"] [data-testid="stMetric"] { border: none; box-shadow: none; background: transparent; padding: 0; }
[class*="st-key-iss_card_det_risk"] [data-testid="stMetricValue"],
[class*="st-key-iss_card_icu_result"] [data-testid="stMetricValue"] { font-size: 3rem; font-weight: 650; line-height: 1.1; }
[class*="st-key-iss_card_det_risk"] .iss-risk { margin-top: 10px; }
/* Flow of stages (flow_steps): e.g. state -> change -> reassessment -> new state. */
.iss-flow { display: flex; align-items: stretch; gap: 6px; }
.iss-flow-step {
    flex: 1 1 0; min-width: 0; border-top: 3px solid var(--iss-border-strong); padding: 8px 12px;
}
.iss-flow-arrow { flex: none; align-self: center; color: var(--iss-subtle); font-size: 1.1rem; }
.iss-flow-title { font-weight: 600; color: var(--iss-ink); font-size: 0.9rem; margin-top: 2px; }
.iss-flow-detail { font-size: 0.8rem; color: var(--iss-muted); margin-top: 3px; line-height: 1.45; overflow-wrap: anywhere; }
.iss-flow .iss-badge-row { margin: 6px 0 0; }
.iss-flow .iss-badge { white-space: normal; text-align: left; max-width: 100%; }
/* Digital Twin overview metrics: let long labels wrap instead of truncating. */
.st-key-twin_overview [data-testid="stMetricLabel"] * { white-space: normal; overflow: visible; }
.iss-flow-lg .iss-flow-step { padding: 12px 14px; box-shadow: var(--iss-shadow-card); }
.iss-flow-lg .iss-flow-title { font-size: 1rem; }
.iss-flow-lg .iss-flow-detail { font-size: 0.84rem; }
@media (max-width: 900px) {
    .iss-flow { flex-direction: column; }
    .iss-flow-arrow { transform: rotate(90deg); }
}

/* Status grid (status_grid) */
.iss-sgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
.iss-sgrid-item { padding: 10px 12px; }
.iss-sgrid-head { display: flex; align-items: center; gap: 8px; }
.iss-sgrid-value { font-weight: 600; color: var(--iss-ink); font-size: 0.92rem; margin-top: 4px; }
.iss-sgrid-detail { font-size: 0.8rem; color: var(--iss-muted); margin-top: 2px; line-height: 1.4; }

"""


def inject_css():
    """Add the global stylesheet to the page (call once per run, from app.py)."""
    st.html(f"<style>{_root_variables()}{_CSS}</style>")
