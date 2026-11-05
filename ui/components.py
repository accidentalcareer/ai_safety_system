"""
Reusable presentation components for the AEGISCARE AI interface.

Each function either draws something (verb-like names: page_header,
section_header, card, ...) or returns an HTML string to embed in another
component (names ending in `_html`). Styling lives in ui/styles.py; this
file only builds markup. No application logic belongs here.

Text passed in is HTML-escaped, so values from the data can be shown safely.
Use the `*_html` arguments only for markup built by these helpers.

Tones: "neutral", "info", "success", "warning", "danger".
Green / yellow / red tones are for risk or range states only.
"""

from html import escape

import streamlit as st

from ui.theme import BRAND_NAME, BRAND_SUFFIX, BRAND_TAGLINE, RISK_LEVELS, TONES


def _text(value):
    """Escape plain text for HTML. `$` is escaped so Markdown never reads it as LaTeX."""
    return escape(str(value)).replace("$", "&#36;")


def _tone(tone):
    if tone not in TONES:
        raise ValueError(f"Unknown tone {tone!r}; expected one of {sorted(TONES)}")
    return tone


def _tone_style(tone, *parts):
    """Inline style using a tone's CSS variables, e.g. parts=("color", "background")."""
    names = {"color": "fg", "background": "bg", "border-color": "border"}
    return ";".join(f"{p}:var(--iss-{_tone(tone)}-{names[p]})" for p in parts)


def _render(markup):
    # Newlines are removed so Markdown never turns indented HTML into a code block.
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Inline pieces (return HTML strings)
# ---------------------------------------------------------------------------

def badge_html(text, tone="neutral", dot=True):
    """A small rounded status label. Always carries text, never colour alone."""
    style = _tone_style(tone, "color", "background", "border-color")
    dot_html = '<span class="iss-dot"></span>' if dot else ""
    return f'<span class="iss-badge" style="{style}">{dot_html}{_text(text)}</span>'


def _as_tags(badges):
    """Badges without a status dot (descriptive tags, e.g. in page headers)."""
    return [(b[0], b[1], False) if isinstance(b, tuple) else (b, "neutral", False) for b in badges or []]


def badges_html(badges):
    """Row of badges from a list of (text, tone) pairs or plain strings."""
    items = [badge_html(*b) if isinstance(b, tuple) else badge_html(b) for b in badges or []]
    return f'<div class="iss-badge-row">{"".join(items)}</div>' if items else ""


def risk_indicator_html(level, label=None, detail=None):
    """Three-step scale plus a text label, e.g. level="moderate"."""
    if level not in RISK_LEVELS:
        raise ValueError(f"Unknown risk level {level!r}; expected one of {sorted(RISK_LEVELS)}")
    tone, default_label, lit = RISK_LEVELS[level]
    lit_style = f"background:var(--iss-{tone}-fg)"
    steps = "".join(
        f'<span class="iss-risk-step" style="{lit_style if i < lit else ""}"></span>' for i in range(3)
    )
    detail_html = f'<span class="iss-risk-detail">{_text(detail)}</span>' if detail else ""
    return (
        f'<div class="iss-risk"><div class="iss-risk-scale">{steps}</div>'
        f'<div class="iss-risk-text"><span class="iss-risk-label" style="{_tone_style(tone, "color")}">'
        f"{_text(label or default_label)}</span>{detail_html}</div></div>"
    )


# ---------------------------------------------------------------------------
# Block components (draw on the page)
# ---------------------------------------------------------------------------

# Shield outline with a cross: the brand mark, drawn in white on navy.
_BRAND_MARK_SVG = (
    '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#FFFFFF" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 3l7 3v5c0 4.5-3 8.2-7 10-4-1.8-7-5.5-7-10V6l7-3z"/><path d="M12 9v6M9 12h6"/></svg>'
)


def sidebar_brand():
    """Brand block at the top of the sidebar."""
    _render(
        f"""
        <div class="iss-brand">
          <div class="iss-brand-mark">{_BRAND_MARK_SVG}</div>
          <div><div class="iss-brand-name">{_text(BRAND_NAME)} <span>{_text(BRAND_SUFFIX)}</span></div>
          <div class="iss-brand-tagline">{_text(BRAND_TAGLINE)}</div></div>
        </div>
        """
    )


def nav_item(text, icon, active, on_click, args, key):
    """
    One sidebar navigation row. The active row is a primary button (styled
    as a soft highlight); the others are tertiary. `key` must start with
    "nav_" so the navigation CSS applies.
    """
    st.button(
        text, icon=icon, key=key, on_click=on_click, args=args,
        type="primary" if active else "tertiary", width="stretch",
    )


def system_status(items):
    """
    Compact status list; `items` is a list of (name, value, tone).
    The tone colours the dot, and the value (shown under the name, so long
    values never squeeze it) always says the state in words.
    """
    rows = "".join(
        f'<div class="iss-status-row"><span class="iss-status-dot" style="background:var(--iss-{_tone(tone)}-fg)">'
        f'</span><span class="iss-status-name">{_text(name)}</span>'
        f'<span class="iss-status-value">{_text(value)}</span></div>'
        for name, value, tone in items
    )
    _render(f'<div class="iss-status">{rows}</div>')


def demo_mode_note(title, text):
    """Dashed note marking the app as a demo (sidebar footer)."""
    _render(f'<div class="iss-demo"><div class="iss-demo-title">{_text(title)}</div>'
            f'<div class="iss-demo-text">{_text(text)}</div></div>')


def label(text):
    """Small uppercase label above a group of controls (e.g. in the sidebar)."""
    _render(f'<div class="iss-label">{_text(text)}</div>')


def page_header(title, subtitle=None, eyebrow=None, badges=None):
    """
    Page title block. The title is a real st.title and the subtitle a real
    st.caption, so headings stay accessible and visible to AppTest. Looks the
    same as page_title() (the keyed container tightens the spacing).
    """
    with st.container(key="iss_page_head"):
        if eyebrow:
            _render(f'<div class="iss-eyebrow">{_text(eyebrow)}</div>')
        st.title(title)
        if subtitle:
            st.caption(subtitle)
        if badges:
            _render(badges_html(_as_tags(badges)))


def page_title(title, subtitle=None, eyebrow=None, badges=None):
    """
    HTML page title, for pages whose st.title is used for something else
    (the Patient Dashboard keeps the patient name as its st.title).
    """
    eyebrow_html = f'<div class="iss-eyebrow">{_text(eyebrow)}</div>' if eyebrow else ""
    sub_html = f'<div class="iss-page-sub">{_text(subtitle)}</div>' if subtitle else ""
    _render(
        f'<div class="iss-page-head">{eyebrow_html}<div class="iss-page-title">{_text(title)}</div>'
        f"{sub_html}{badges_html(_as_tags(badges))}</div>"
    )


def section_header(title, description=None, step=None, badges=None):
    """Section title with an optional step number, description and badges."""
    step_html = f'<div class="iss-section-step">{_text(step)}</div>' if step is not None else ""
    desc_html = f'<div class="iss-section-desc">{_text(description)}</div>' if description else ""
    _render(
        f'<div class="iss-section">{step_html}<div>'
        f'<div class="iss-section-title">{_text(title)}</div>{desc_html}{badges_html(badges)}'
        "</div></div>"
    )


def card(body=None, title=None, footer=None, accent=False, body_html=None):
    """A white card holding text. Use card_container() for cards with widgets."""
    title_html = f'<div class="iss-card-title">{_text(title)}</div>' if title else ""
    content = body_html if body_html is not None else (_text(body) if body else "")
    footer_html = f'<div class="iss-card-footer">{_text(footer)}</div>' if footer else ""
    classes = "iss-card iss-card-accent" if accent else "iss-card"
    _render(
        f'<div class="{classes}">{title_html}<div class="iss-card-body">{content}</div>{footer_html}</div>'
    )


def card_container(key, tone=None, height=None, fill_height=False, edge="top", muted=False):
    """
    A native Streamlit container styled as a card, for placing widgets,
    charts and tables inside a card. Use as `with card_container("labs"): ...`.
    `tone` adds a thin coloured edge, on the top or (edge="left", for alert
    cards) the left side; `height` (px) makes it scroll; `fill_height`
    stretches the card to its column so cards in a row line up (a button
    placed last sits at the bottom). `muted` marks supporting information
    (explanations, model details): a flat tinted panel without a shadow.
    """
    suffix = (f"__{_tone(tone)}" if tone else "") + ("__left" if tone and edge == "left" else "")
    suffix += "__fill" if fill_height else ""
    suffix += "__muted" if muted else ""
    kwargs = {"height": height} if height else {}
    return st.container(key=f"iss_card_{key}{suffix}", **kwargs)


def summary_block(label, headline, detail=None, rows=None, badges=None, tone=None):
    """
    Text content for a summary card: small label, bold headline, optional
    detail line, (name, value) rows and badges. `tone` colours the headline.
    """
    headline_style = f' style="{_tone_style(tone, "color")}"' if tone else ""
    detail_html = f'<div class="iss-summary-detail">{_text(detail)}</div>' if detail else ""
    rows_html = "".join(
        f'<div class="iss-summary-row"><span>{_text(name)}</span><span>{_text(value)}</span></div>'
        for name, value in rows or []
    )
    rows_html = f'<div class="iss-summary-rows">{rows_html}</div>' if rows_html else ""
    _render(
        f'<div class="iss-summary"><div class="iss-label">{_text(label)}</div>'
        f'<div class="iss-summary-headline"{headline_style}>{_text(headline)}</div>'
        f"{detail_html}{badges_html(badges)}{rows_html}</div>"
    )


def metric_card_html(label, value, unit=None, status=None, status_tone="neutral", caption=None):
    """HTML for one metric card. The left edge takes the status tone colour."""
    unit_html = f'<span class="iss-metric-unit">{_text(unit)}</span>' if unit else ""
    status_html = badge_html(status, status_tone) if status else ""
    caption_html = f"<span>{_text(caption)}</span>" if caption else ""
    foot = f'<div class="iss-metric-foot">{status_html}{caption_html}</div>' if status or caption else ""
    edge = f"border-left-color:var(--iss-{_tone(status_tone)}-fg)" if status and status_tone != "neutral" else ""
    return (
        f'<div class="iss-metric" style="{edge}"><div class="iss-metric-label">{_text(label)}</div>'
        f'<div class="iss-metric-value">{_text(value)}{unit_html}</div>{foot}</div>'
    )


def metric_card(label, value, unit=None, status=None, status_tone="neutral", caption=None):
    """
    A display-only metric card. Not a st.metric, so it does not count as a
    Streamlit metric element (pages whose tests count st.metric stay valid).
    """
    _render(metric_card_html(label, value, unit, status, status_tone, caption))


def metric_grid(cards):
    """Responsive grid of metric cards; `cards` is a list of metric_card_html() strings."""
    _render(f'<div class="iss-metric-grid">{"".join(cards)}</div>')


def badge_row(badges):
    """Draw a row of badges; items are (text, tone[, dot]) tuples or plain strings."""
    _render(badges_html(badges))


def status_badge(text, tone="neutral"):
    """Draw a single status badge on its own line."""
    _render(badge_html(text, tone))


def risk_indicator(level, label=None, detail=None):
    """Draw a three-step risk indicator: "low", "moderate", "high" or "unknown"."""
    _render(risk_indicator_html(level, label, detail))


def info_panel(body, title=None, tone="info"):
    """Calm callout with a coloured edge. For notes that are not errors."""
    title_html = (
        f'<div class="iss-panel-title" style="{_tone_style(tone, "color")}">{_text(title)}</div>' if title else ""
    )
    _render(
        f'<div class="iss-panel" style="{_tone_style(tone, "color", "background", "border-color")}">'
        f'<div class="iss-panel-bar"></div><div>{title_html}<div class="iss-panel-body">{_text(body)}</div></div></div>'
    )


def tile_html(title, subtitle=None, detail=None, badges=None, tone=None):
    """HTML for one compact tile (e.g. a medication); `tone` colours its left edge."""
    edge = f' style="border-left-color:var(--iss-{_tone(tone)}-fg)"' if tone else ""
    sub_html = f'<div class="iss-tile-sub">{_text(subtitle)}</div>' if subtitle else ""
    detail_html = f'<div class="iss-tile-detail">{_text(detail)}</div>' if detail else ""
    return (
        f'<div class="iss-tile"{edge}><div class="iss-tile-title">{_text(title)}</div>'
        f"{sub_html}{detail_html}{badges_html(badges)}</div>"
    )


def tile_grid(tiles):
    """Responsive grid of tile_html() strings."""
    _render(f'<div class="iss-tile-grid">{"".join(tiles)}</div>')


def fact_rows(rows):
    """
    Small three-column list of recorded facts; `rows` is a list of
    (item, value, note, missing) where missing=True shows the value muted.
    """
    cells = "".join(
        f'<div class="iss-fact"><span class="iss-fact-item">{_text(item)}</span>'
        f'<span class="iss-fact-value{" iss-fact-missing" if missing else ""}">{_text(value)}</span>'
        f'<span class="iss-fact-note">{_text(note)}</span></div>'
        for item, value, note, missing in rows
    )
    _render(f'<div class="iss-facts">{cells}</div>')


def key_value_grid(items):
    """Labelled values in a responsive grid; `items` is a list of (label, value)."""
    cells = "".join(
        f'<div><div class="iss-kv-label">{_text(k)}</div><div class="iss-kv-value">{_text(v)}</div></div>'
        for k, v in items
    )
    _render(f'<div class="iss-kv">{cells}</div>')


def timeline_entry_html(title, time=None, body=None, meta=None, tone="neutral", badges=None):
    """HTML for one timeline entry; combine several with timeline()."""
    dot_style = f"border-color:var(--iss-{_tone(tone)}-fg)"
    time_html = f'<span class="iss-tl-time">{_text(time)}</span>' if time else ""
    body_html = f'<div class="iss-tl-body">{_text(body)}</div>' if body else ""
    meta_html = f'<div class="iss-tl-meta">{_text(meta)}</div>' if meta else ""
    return (
        f'<div class="iss-tl-entry"><div class="iss-tl-dot" style="{dot_style}"></div><div>'
        f'<div class="iss-tl-head"><span class="iss-tl-title">{_text(title)}</span>{time_html}</div>'
        f"{badges_html(badges)}{body_html}{meta_html}</div></div>"
    )


def timeline(entries):
    """Vertical timeline from a list of timeline_entry_html() strings."""
    _render(f'<div class="iss-timeline">{"".join(entries)}</div>')


def mode_strip(items):
    """
    Row of mutually exclusive modes with the current one highlighted, e.g.
    which kind of output a result is. `items` is a list of
    (title, detail, active, tone); only the active item takes its tone colour.
    """
    cells = []
    for title, detail, active, tone in items:
        if active:
            attrs = f' class="iss-mode-item is-active" style="{_tone_style(tone, "color", "background", "border-color")}"'
        else:
            attrs = ' class="iss-mode-item"'
        cells.append(
            f'<div{attrs}><div class="iss-mode-state">{"Current result" if active else "Not in use"}</div>'
            f'<div class="iss-mode-title">{_text(title)}</div><div class="iss-mode-detail">{_text(detail)}</div></div>'
        )
    _render(f'<div class="iss-mode">{"".join(cells)}</div>')


def flow_steps(steps, prominent=False):
    """
    Left-to-right sequence of stages joined by arrows; `steps` is a list of
    (label, title, detail, badges, tone). A tone colours that stage's top
    edge; `prominent` gives larger cards (e.g. a state transition).
    """
    cells = []
    for i, (step_label, title, detail, badges, tone) in enumerate(steps):
        edge = f' style="border-top-color:var(--iss-{_tone(tone)}-fg)"' if tone else ""
        detail_html = f'<div class="iss-flow-detail">{_text(detail)}</div>' if detail else ""
        if i:
            cells.append('<div class="iss-flow-arrow" aria-hidden="true">&rarr;</div>')
        cells.append(
            f'<div class="iss-flow-step"{edge}><div class="iss-flow-label">{_text(step_label)}</div>'
            f'<div class="iss-flow-title">{_text(title)}</div>{detail_html}{badges_html(badges)}</div>'
        )
    _render(f'<div class="iss-flow{" iss-flow-lg" if prominent else ""}">{"".join(cells)}</div>')


def status_grid(items):
    """
    Grid of subtle status tiles; `items` is a list of (name, value, detail, tone).
    The tone colours only the dot; the value always states the status in words.
    """
    cells = "".join(
        f'<div class="iss-sgrid-item"><div class="iss-sgrid-head"><span class="iss-status-dot" '
        f'style="background:var(--iss-{_tone(tone)}-fg)"></span><span class="iss-sgrid-name">{_text(name)}</span></div>'
        f'<div class="iss-sgrid-value">{_text(value)}</div>'
        + (f'<div class="iss-sgrid-detail">{_text(detail)}</div>' if detail else "")
        + "</div>"
        for name, value, detail, tone in items
    )
    _render(f'<div class="iss-sgrid">{cells}</div>')
