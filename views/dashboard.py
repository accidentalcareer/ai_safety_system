"""
Patient Dashboard page – AEGISCARE AI landing page and Patient Safety Overview.

Opens with what the application is: a short hero, one card per module (with
whether it uses documented rules, a simulated score, a trained model or a
simulation) and a compact system status. Below that, a safety summary for the
selected synthetic patient: identity,
three safety summaries, the latest vital signs with mini-trends, one large
trend chart, recent safety events, the reasons behind the summaries, and
the patient's medications and laboratory results.

Every value shown comes from the existing data and modules (range checks,
Drug Interaction Checker, Deterioration demo score, Digital Twin timeline).
This page adds no clinical rule, threshold or interpretation of its own.
"""

import pandas as pd
import streamlit as st

from data.drug_interaction_rules import INTERACTION_RULES
from data.synthetic_patients import (
    HOURS_OF_HISTORY,
    PATIENTS,
    READING_INTERVAL_HOURS,
    VITAL_SIGNS,
    classify_value,
    generate_vitals,
    get_lab_results,
    get_medications,
    get_patient,
)
from ui.components import (
    badge_row,
    card_container,
    key_value_grid,
    label,
    page_title,
    section_header,
    status_grid,
    summary_block,
    timeline,
    timeline_entry_html,
)
from utils.charts import blood_pressure_chart, contributions_chart, vital_trend_chart
from utils.deterioration import MAX_PART_POINTS, TREND_WINDOW_HOURS, compute_demo_score
from utils.drug_interactions import KNOWN_DRUGS, check_interactions, display_name, patient_medication_names
from views.icu_risk import model_status

# Text + icon for each range status, so meaning never depends on colour alone.
STATUS_LABELS = {
    "Low": "↓ Below typical range",
    "High": "↑ Above typical range",
    "Normal": "✓ Within typical range",
}
LAB_STATUS_LABELS = {"Low": "↓ Low", "High": "↑ High", "Normal": "✓ Normal"}

# Vitals that get their own single-series trend chart (BP has a combined chart).
SINGLE_VITAL_CHARTS = ["heart_rate", "spo2", "resp_rate", "temperature"]

# Short status text for the compact vital cards; the arrow shows the direction.
STATUS_SHORT = {"Low": "Below range", "High": "Above range", "Normal": "In range"}
STATUS_ARROW = {"Low": "down", "High": "up", "Normal": "off"}
STATUS_ICON_SHORT = {"Low": "↓ Below range", "High": "↑ Above range", "Normal": "✓ In range"}

# Order of the vital cards and trend selector; "blood_pressure" combines systolic/diastolic.
VITAL_CARDS = ["heart_rate", "blood_pressure", "spo2", "resp_rate", "temperature"]
TREND_NAMES = {"blood_pressure": "Blood pressure"} | {k: VITAL_SIGNS[k]["short"] for k in SINGLE_VITAL_CHARTS}

# Readings in the last 24 h (mini-trends and range alerts).
READINGS_PER_DAY = 24 // READING_INTERVAL_HOURS + 1

# Module cards: (page, icon, title, description, (result-type badge, tone), button key, action label).
CAPABILITIES = [
    ("Drug Interaction Checker", ":material/medication:", "Drug interaction analysis",
     "Checks a medication list against documented interaction rules and explains each match with its source.",
     ("Documented rules", "info"), "open_interaction_checker", "Check interactions"),
    ("Deterioration Prediction", ":material/monitor_heart:", "Patient safety monitoring",
     "Vital-sign trends and a transparent deviation score for the synthetic patients.",
     ("Simulated score", "neutral"), "open_deterioration_page", "Open monitoring"),
    ("ICU Risk Assessment", ":material/ecg_heart:", "ICU risk assessment",
     "Model-derived ICU mortality risk score from first-24-hour clinical measurements.",
     ("Trained model", "info"), "open_icu_risk", "Assess ICU risk"),
    ("Patient Digital Twin", ":material/hub:", "Patient digital twin",
     "Simulate medication or vital-sign changes on a copy of a patient and watch the modules reassess.",
     ("Simulation", "neutral"), "open_digital_twin", "Open digital twin"),
]

# Twin registry key in session state (same value as views.digital_twin.REGISTRY_KEY;
# not imported because views.digital_twin imports this module indirectly).
TWIN_REGISTRY_KEY = "twin_registry"


@st.cache_data
def load_vitals(patient_id):
    """Cache the generated vitals so the page stays fast."""
    return generate_vitals(patient_id)


@st.cache_data
def _cohort_demo_scores():
    """Demo scores of all synthetic patients, for the rank shown on the score card."""
    return {p["patient_id"]: compute_demo_score(load_vitals(p["patient_id"]), p).score for p in PATIENTS}


def render(patient_id):
    """Draw the whole dashboard for one patient."""
    patient = get_patient(patient_id)
    vitals = load_vitals(patient_id)
    labs = get_lab_results(patient_id)
    interactions = check_interactions(patient_medication_names(patient))
    demo = compute_demo_score(vitals, patient)

    _render_hero()
    _render_capabilities()
    _render_engine_status()

    _render_header(patient, vitals)
    _render_identity(patient)
    _render_safety_cards(patient_id, vitals.iloc[-1], labs, interactions, demo)
    _render_latest_vitals(vitals)

    chart_col, events_col = st.container(key="iss_wrap_trend").columns([2, 1], gap="medium")
    with chart_col:
        _render_trends(vitals)
    with events_col:
        _render_events(vitals, patient_id)

    _render_patient_information(patient, patient_id, labs, interactions)
    _render_risk_explanation(demo, interactions)


def _go_to_page(page_name):
    """Button callback: switch the sidebar navigation to another page."""
    st.session_state["page"] = page_name


# ---------------------------------------------------------------------------
# 0. Landing: hero, modules, system status
# ---------------------------------------------------------------------------

def _render_hero():
    page_title(
        "AEGISCARE AI",
        subtitle=(
            "Integrated healthcare-safety intelligence: medication interaction checks, deterioration "
            "monitoring, ICU risk modelling and patient simulation in one place."
        ),
        eyebrow="Academic demonstration",
        badges=[("Synthetic patients + eICU demo data", "neutral"), ("Not a medical device", "neutral")],
    )
    st.button("Start an ICU risk assessment", type="primary", icon=":material/arrow_forward:",
              key="hero_open_icu", on_click=_go_to_page, args=("ICU Risk Assessment",))


def _capability_card(page, icon, title, text, badge, key, action, detail):
    with card_container(f"cap_{key}", fill_height=True):
        st.markdown(f"#### {icon} {title}")
        badge_row([(badge[0], badge[1], False)])
        st.markdown(text)
        st.caption(detail)
        st.button(action, icon=":material/arrow_forward:", key=key, type="tertiary",
                  on_click=_go_to_page, args=(page,))


def _render_capabilities():
    section_header("Modules", "What each part of AEGISCARE AI does, and what powers it.")
    icu_text, _ = model_status()
    details = {
        "Drug Interaction Checker": f"{len(INTERACTION_RULES)} sample rules · {len(KNOWN_DRUGS)} recognised drugs",
        "Deterioration Prediction": "Arithmetic score on illustrative ranges · not a trained model",
        "ICU Risk Assessment": f"{icu_text} · eICU demo data",
        "Patient Digital Twin": "Edits a simulated copy; the original record is unchanged",
    }
    for start in (0, 2):  # 2 x 2 grid; stacks in order on narrow screens
        for col, card in zip(st.columns(2, gap="medium"), CAPABILITIES[start:start + 2]):
            with col:
                _capability_card(*card, details[card[0]])


def _render_engine_status():
    icu_text, icu_tone = model_status()
    icu_state, _, icu_model = icu_text.partition(" · ")
    label("System status")
    status_grid([
        ("Interaction rules", "Documented", f"{len(INTERACTION_RULES)} rules from official product labelling", "info"),
        ("Deterioration engine", "Simulated", "Deviation score on synthetic patients", "neutral"),
        ("ICU risk model", icu_state,
         f"{icu_model} · eICU demo data" if icu_model else "Model files could not be loaded", icu_tone),
        ("Digital twin", "Simulation", "Reuses the rules and the simulated score", "neutral"),
    ])


# ---------------------------------------------------------------------------
# 1–2. Selected patient: header and identity
# ---------------------------------------------------------------------------

def _render_header(patient, vitals):
    latest_time = vitals["timestamp"].iloc[-1]
    section_header(
        "Selected patient",
        "Safety overview of the synthetic patient chosen in the sidebar: medication interactions, "
        "deterioration signals and vital-sign trends.",
        badges=[
            (f"Selected: {patient['display_name']}", "info", False),
            ("Synthetic data", "neutral", False),
            (f"Vitals every {READING_INTERVAL_HOURS} h · {len(vitals)} readings / {HOURS_OF_HISTORY} h",
             "neutral", False),
            (f"Last reading {latest_time:%d %b %Y, %H:%M}", "neutral", False),
        ],
    )


def _render_identity(patient):
    with card_container("identity"):
        name_col, details_col = st.columns([2, 3], vertical_alignment="center")
        with name_col:
            st.title(patient["display_name"])
            st.caption(f"Patient ID **{patient['patient_id']}** · Synthetic record for demonstration only.")
        with details_col:
            key_value_grid([
                ("Age / Sex", f"{patient['age']} / {patient['sex']}"),
                ("Ward", patient["ward"]),
                ("Admitted", patient["admission_date"]),
                ("Attending", patient["attending"]),
            ])
        allergies = [a for a in patient["allergies"] if a.lower() != "none known"]
        allergy_badges = (
            [(f"Allergy: {a}", "danger", False) for a in allergies]
            if allergies else [("No known allergies", "neutral", False)]
        )
        badge_row(allergy_badges + [(d, "neutral", False) for d in patient["diagnoses"]])
        st.caption(f"Scenario: {patient['scenario']}")


# ---------------------------------------------------------------------------
# 3. Safety summary cards
# ---------------------------------------------------------------------------

def _render_safety_cards(patient_id, latest, labs, interactions, demo):
    status_col, drug_col, det_col = st.container(key="iss_wrap_safety").columns(3, gap="medium")
    with status_col:
        _patient_status_card(latest, labs)
    with drug_col:
        _interaction_card(interactions)
    with det_col:
        _deterioration_card(patient_id, demo)


def _patient_status_card(latest, labs):
    out_of_range = []
    for key, info in VITAL_SIGNS.items():
        status = classify_value(latest[key], info["low"], info["high"])
        if status != "Normal":
            out_of_range.append((info["short"], f"{latest[key]:g} {info['unit']} · {STATUS_ICON_SHORT[status]}"))
    labs_out = int((labs["Status"] != "Normal").sum())

    if out_of_range:
        tone = "warning"
        headline = f"{len(out_of_range)} of {len(VITAL_SIGNS)} latest vitals outside typical range"
    else:
        tone = "success"
        headline = "All latest vitals within typical range"
    with card_container("safety_status", tone=tone, fill_height=True):
        summary_block(
            "Patient status",
            headline,
            detail=f"{labs_out} of {len(labs)} lab results outside the reference range shown.",
            rows=out_of_range[:3],
            badges=[("Range comparison, not a diagnosis", "neutral", False)],
            tone=tone,
        )


def _interaction_card(interactions):
    count = len(interactions.interactions)
    if count:
        tone = "warning"
        headline = f"{count} documented interaction{'s' if count != 1 else ''} matched"
        detail = "Matched against the sample rules taken from official product labelling."
    else:
        tone = "neutral"
        headline = "No documented interaction matched"
        detail = f"Only {len(INTERACTION_RULES)} sample rules are checked, so this does not mean the combination is safe."
    if interactions.unknown_drugs:
        detail += " Not checked: " + ", ".join(interactions.unknown_drugs) + "."
    with card_container("safety_interactions", tone=tone, fill_height=True):
        summary_block(
            "Drug interaction risk",
            headline,
            detail=detail,
            rows=[(f"{display_name(i.drug_a)} + {display_name(i.drug_b)}", i.concern) for i in interactions.interactions[:3]],
            badges=[("Documented rule", "info", False)],
            tone=tone if count else None,
        )


def _deterioration_card(patient_id, demo):
    cohort = _cohort_demo_scores()
    # Same ranking as the Deterioration page: 1 = highest demo score.
    rank = sum(score > demo.score for pid, score in cohort.items() if pid != patient_id) + 1
    factors = [(c.label, f"{c.points:g} points") for c in demo.contributions if c.points > 0]
    with card_container("safety_deterioration", tone="neutral", fill_height=True):
        summary_block(
            "Deterioration risk",
            f"Demo deviation score {demo.score:g}",
            detail=(
                f"Rank {rank} of {len(cohort)} synthetic patients (1 = highest) · "
                f"{demo.items_used} of {demo.items_expected} inputs available."
            ),
            rows=factors[:3] or [("Contributing factors", "none – all inputs within range")],
            badges=[("Simulated score, not a probability", "neutral", False),
                    ("No risk band: no validated cut-off", "neutral", False)],
        )


# ---------------------------------------------------------------------------
# 4. Latest vital signs with mini-trends
# ---------------------------------------------------------------------------

def _render_latest_vitals(vitals):
    latest = vitals.iloc[-1]
    section_header(
        "Latest vital signs",
        f"Recorded {latest['timestamp']:%d %b %Y, %H:%M}. Change is against the previous reading "
        f"({READING_INTERVAL_HOURS} hours earlier); mini-trends show the last 24 h.",
    )
    render_vital_metrics(vitals)


# Fixed card width: five fit across a desktop page, and the row wraps (2 across on a
# phone) instead of shrinking the cards until their values are cut off.
VITAL_CARD_WIDTH = 170


def render_vital_metrics(vitals):
    """Latest-reading card per vital sign with a 24 h mini-trend (also used by the Deterioration page)."""
    latest = vitals.iloc[-1]
    previous = vitals.iloc[-2]
    recent = vitals.tail(READINGS_PER_DAY)
    row = st.container(horizontal=True, horizontal_alignment="distribute", gap="small")
    for key in VITAL_CARDS:
        # Blood pressure is shown as one card: systolic/diastolic, status from systolic.
        series = "systolic_bp" if key == "blood_pressure" else key
        info = VITAL_SIGNS[series]
        decimals = 1 if key == "temperature" else 0
        status = classify_value(latest[series], info["low"], info["high"])
        if key == "blood_pressure":
            label_text = "BP (mmHg)"
            value = f"{latest['systolic_bp']:.0f}/{latest['diastolic_bp']:.0f}"
            status_text = STATUS_SHORT[status]  # from systolic; the help text says so
            change = f"{latest[series] - previous[series]:+.0f} sys."
        else:
            label_text = f"{info['short']} ({info['unit']})"
            value = f"{latest[key]:.{decimals}f}"
            status_text = STATUS_SHORT[status]
            change = f"{latest[key] - previous[key]:+.{decimals}f}"
        row.metric(
            label_text,
            value,
            delta=status_text,
            delta_color="gray" if status == "Normal" else "yellow",
            delta_arrow=STATUS_ARROW[status],
            delta_description=change,
            chart_data=recent[series].tolist(),
            chart_type="line",
            border=True,
            width=VITAL_CARD_WIDTH,
            help=f"{'Systolic: ' if key == 'blood_pressure' else ''}{STATUS_LABELS[status]} (typical adult range {info['low']:g}–{info['high']:g} "
                 f"{info['unit']}). A simple range comparison, not a clinical interpretation.",
        )


# ---------------------------------------------------------------------------
# 5. Main trend chart
# ---------------------------------------------------------------------------

def _render_trends(vitals):
    with card_container("trend"):
        title_col, window_col = st.container(key="iss_wrap_dash_trend_head").columns([3, 2], vertical_alignment="center")
        with title_col:
            label("Physiological trends")
        with window_col, st.container(horizontal=True, horizontal_alignment="right"):
            window = st.segmented_control(
                "Time window (most recent)", [24, 48, 72], default=72, format_func=lambda h: f"{h} h",
                key="dash_trend_window", label_visibility="collapsed",
            ) or 72
        selected = st.segmented_control(
            "Vital sign", VITAL_CARDS, default="heart_rate", format_func=TREND_NAMES.get,
            key="dash_trend_vital", label_visibility="collapsed",
        ) or "heart_rate"

        cutoff = vitals["timestamp"].max() - pd.Timedelta(hours=window)
        window_df = vitals[vitals["timestamp"] >= cutoff]
        fig = blood_pressure_chart(window_df) if selected == "blood_pressure" else vital_trend_chart(window_df, selected)
        fig.update_layout(height=330, title_text="", margin=dict(t=30))
        st.plotly_chart(fig, width="stretch", key="dash_trend_chart")
        st.caption(
            "Shaded bands show typical adult reference ranges for visual context only. "
            "They are not patient-specific targets."
        )

        with st.expander("Show raw vital-sign readings"):
            table = window_df.rename(
                columns={key: f"{info['label']} ({info['unit']})" for key, info in VITAL_SIGNS.items()}
            ).rename(columns={"timestamp": "Time"})
            st.dataframe(table.sort_values("Time", ascending=False), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# 7. Recent safety events
# ---------------------------------------------------------------------------

def _range_alert_entries(vitals):
    """One entry per vital sign with readings outside its typical range in the last 24 h."""
    recent = vitals.tail(READINGS_PER_DAY)
    alerts = []
    for key, info in VITAL_SIGNS.items():
        statuses = recent[key].map(lambda v: classify_value(v, info["low"], info["high"]))
        flagged = recent[statuses != "Normal"]
        if flagged.empty:
            continue
        last = flagged.iloc[-1]
        direction = "below" if statuses.loc[last.name] == "Low" else "above"
        alerts.append((last["timestamp"], timeline_entry_html(
            f"{info['label']} {direction} typical range",
            time=f"{last['timestamp']:%d %b %H:%M}",
            body=f"{len(flagged)} of {len(recent)} readings in the last 24 h · latest flagged {last[key]:g} {info['unit']}",
            tone="warning",
        )))
    return [entry for _, entry in sorted(alerts, key=lambda a: a[0], reverse=True)]


def _twin_entries(patient_id, limit=3):
    """Newest Digital Twin changes for this patient made in this session (if any)."""
    registry = st.session_state.get(TWIN_REGISTRY_KEY)
    if registry is None or patient_id not in registry:
        return []
    events = [e for e in registry.get(patient_id).events if e.change_type != "Twin created"]
    return [
        timeline_entry_html(
            f"{e.change_type}: {e.item}",
            time=f"{e.timestamp:%H:%M:%S}",
            body=f"{e.previous_value} → {e.new_value}",
            meta=f"Resulting status: {e.status}",
            tone="info",
        )
        for e in reversed(events[-limit:])
    ]


def _render_events(vitals, patient_id):
    with card_container("events", height=540):
        label("Recent safety events")
        st.caption("Synthetic record · last 24 h")
        alerts = _range_alert_entries(vitals)
        if alerts:
            timeline(alerts)
        else:
            st.caption("No readings outside the typical ranges in the last 24 h.")

        st.caption("Patient Digital Twin · this session (simulated)")
        twin = _twin_entries(patient_id)
        if twin:
            timeline(twin)
        else:
            st.caption("No simulated changes yet. Changes made on the Patient Digital Twin page appear here.")


# ---------------------------------------------------------------------------
# 6. Patient information
# ---------------------------------------------------------------------------

def _render_patient_information(patient, patient_id, labs, interactions):
    section_header("Patient information", "Medications, laboratory results and admission details from the record.")
    with card_container("patient_info"):
        meds_tab, labs_tab, details_tab = st.tabs(
            ["Current medications", "Laboratory results", "Demographics & admission"]
        )
        with meds_tab:
            _render_medications(patient_id, interactions)
        with labs_tab:
            _render_labs(labs)
        with details_tab:
            key_value_grid([
                ("Patient ID", patient["patient_id"]),
                ("Age", patient["age"]),
                ("Sex", patient["sex"]),
                ("Ward", patient["ward"]),
                ("Admission date", patient["admission_date"]),
                ("Attending", patient["attending"]),
                ("Diagnoses", ", ".join(patient["diagnoses"])),
                ("Allergies", ", ".join(patient["allergies"])),
            ])
            st.caption(f"Scenario: {patient['scenario']}")


def _render_medications(patient_id, interactions):
    meds = get_medications(patient_id)
    st.markdown(f"**{len(meds)} active medications** (synthetic list)")
    st.dataframe(meds, hide_index=True, width="stretch")
    if interactions.unknown_drugs:
        st.caption("Not in the sample dataset (not checked): " + ", ".join(interactions.unknown_drugs))


def _render_labs(labs):
    out_of_range = (labs["Status"] != "Normal").sum()
    st.markdown(f"**{out_of_range} of {len(labs)}** results outside the reference range shown.")

    display = labs.copy()
    display["Status"] = display["Status"].map(LAB_STATUS_LABELS)
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(
        "Reference ranges are illustrative adult ranges; real ranges vary by laboratory. "
        "The status column is a simple range comparison, not a clinical interpretation."
    )


# ---------------------------------------------------------------------------
# 8. Why this risk?
# ---------------------------------------------------------------------------

def _render_risk_explanation(demo, interactions):
    section_header(
        "Why this risk?",
        "The reasons recorded by the modules themselves: the demo score's factor points and the "
        "documented interaction rules. No further interpretation is added.",
    )
    det_col, drug_col = st.columns(2, gap="medium")
    with det_col, card_container("why_deterioration", muted=True):
        label("Deterioration demo score")
        if any(c.points > 0 for c in demo.contributions):
            fig = contributions_chart(demo.contributions)
            fig.update_layout(title_text="", margin=dict(t=10))
            # Headroom so the value labels outside the bars are not clipped.
            fig.update_xaxes(range=[0, max(c.points for c in demo.contributions) * 1.25])
            st.plotly_chart(fig, width="stretch", key="dash_contrib_chart")
        else:
            st.caption("All available inputs are within their illustrative ranges, so no factor adds points. "
                       "This does not mean the patient is safe.")
        st.caption(
            f"Points = distance outside the range (capped at {MAX_PART_POINTS:g}) + worsening over the last "
            f"{TREND_WINDOW_HOURS} h (capped at {MAX_PART_POINTS:g}). Simulated, not a probability."
        )
        if demo.missing:
            with st.expander(f"Missing-data notes ({len(demo.missing)})"):
                st.markdown("\n".join(f"- {note}" for note in demo.missing))

    with drug_col, card_container("why_interactions", muted=True):
        label("Documented interaction rules")
        if not interactions.interactions:
            st.caption(
                f"No documented rule matched. The sample covers only {len(INTERACTION_RULES)} pairs, "
                "so this does not mean the combination is safe."
            )
        for interaction in interactions.interactions:
            st.markdown(
                f"**{display_name(interaction.drug_a)} + {display_name(interaction.drug_b)}** · {interaction.concern}"
            )
            st.markdown(interaction.explanation)
            st.caption("Source: " + "; ".join(
                f"[{s['title']}]({s['url']}) – {s['section']}" for s in interaction.sources
            ))
