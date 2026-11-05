"""
Deterioration Prediction page – Patient Deterioration Intelligence.

Shows the selected synthetic patient's current demo score, latest vital
signs and lab inputs, a selectable trend chart, the factor breakdown of
the score, what kind of model produced it, and its limitations.

No machine-learning model is trained, because the synthetic records have
no outcome labels (see utils/deterioration.py). The score shown here is a
SIMULATED DEMONSTRATION: not a trained-model prediction and not a clinical
probability. No risk band is shown because there is no validated cut-off.
"""

import numpy as np
import pandas as pd
import streamlit as st

from data.synthetic_patients import PATIENTS, VITAL_SIGNS, get_patient
from ui.components import (
    badge_row,
    card_container,
    info_panel,
    key_value_grid,
    label,
    mode_strip,
    page_header,
    risk_indicator,
    section_header,
)
from utils.charts import (
    blood_pressure_chart,
    contribution_breakdown_chart,
    lab_range_chart,
    score_history_chart,
    vital_trend_chart,
)
from utils.deterioration import (
    DEMO_LABS,
    MAX_PART_POINTS,
    OUTCOME_LABEL_KEY,
    RESULT_TYPE_SIMULATED,
    SCORE_VITALS,
    TREND_WINDOW_HOURS,
    USES_MEDICATION_DATA,
    compute_demo_score,
    contributions_table,
    lab_inputs,
    labeled_data_check,
    score_history,
)
from views.dashboard import (
    TREND_NAMES,
    VITAL_CARDS,
    load_vitals,
    render_vital_metrics,
)

# Extra option in the trend selector: the vital-sign part of the demo score.
SCORE_TREND = "demo_score"
TREND_OPTIONS = VITAL_CARDS + [SCORE_TREND]
TREND_LABELS = TREND_NAMES | {SCORE_TREND: "Demo score"}


def render(patient_id):
    """Draw the whole Deterioration Prediction page for one patient."""
    patient = get_patient(patient_id)
    vitals = load_vitals(patient_id)

    page_header(
        "Patient Deterioration Intelligence",
        subtitle=(
            f"Simulated deviation score from the synthetic observations · {patient['display_name']} · "
            f"Patient ID **{patient['patient_id']}** · Synthetic data, educational demo only."
        ),
        eyebrow="Patient safety",
        badges=[(RESULT_TYPE_SIMULATED, "info"), ("Not a probability", "neutral"), ("Synthetic data", "neutral")],
    )
    _render_result_status()

    # Section 1: the risk card shows the score for the (possibly edited) readings,
    # which come from the demo widgets placed below it, so it is filled in afterwards.
    section_header("Current risk", "The current output of the deterioration module.", step=1)
    risk_slot = st.container()
    scored_vitals = _render_what_if(vitals, patient_id)
    result = compute_demo_score(scored_vitals, patient)
    with risk_slot:
        _render_current_risk(result, patient_id)
    _render_data_notes(result)

    section_header(
        "Vital signs and lab inputs",
        "The recorded observations the score reads. Demo edits change only the score, not these values.",
        step=2,
    )
    _render_vital_cards(vitals)
    _render_labs(patient)

    section_header("Trend analysis", "Historical observations for the selected measure.", step=3)
    _render_trends(scored_vitals)

    section_header("Risk contributors", "How much each input added to the current demo score.", step=4)
    _render_contributors(result)

    section_header("Model information", "What produced this result, taken from the implementation.", step=5)
    _render_model_info()

    _render_limitations()


# ---------------------------------------------------------------------------
# Header: result status
# ---------------------------------------------------------------------------

def _render_result_status():
    available, reason = labeled_data_check()
    with st.container(key="iss_notice_det"):
        mode_strip([
            ("Trained model output",
             "Not available for this module – " + ("labelled data found, but no model is implemented yet."
                                                   if available else "the synthetic records have no outcome labels."),
             False, "info"),
            ("Rule-based output",
             "Not used – no validated clinical scoring rule is implemented.", False, "info"),
            ("Simulated demonstration",
             "Active – a transparent arithmetic score on illustrative ranges.", True, "info"),
        ])
        if available:  # not reachable with the current synthetic data
            st.info(f"Labelled data found: {reason} A trained model is not implemented yet.")
        st.error(
            "**Simulated demonstration mode – not a trained model and not a clinical "
            "probability.** The score below is a transparent arithmetic summary of how far "
            "the synthetic readings sit outside illustrative ranges. It does not predict "
            "what will happen to anyone and must not be used for care decisions. "
            "This module uses a simulated score. A separately trained model is available on the ICU Risk Assessment page.",
            icon=":material/science:",
        )


# ---------------------------------------------------------------------------
# 1. Current risk
# ---------------------------------------------------------------------------

@st.cache_data
def _cohort_scores():
    """Demo scores of all synthetic patients (unedited data) for comparison."""
    return {
        p["patient_id"]: compute_demo_score(load_vitals(p["patient_id"]), p).score
        for p in PATIENTS
    }


def _render_current_risk(result, patient_id):
    with card_container("det_risk", tone="neutral"):
        if not result.contributions:
            label("Demo deviation score")
            st.info("Not enough valid data to calculate a demo score.")
            return

        cohort = _cohort_scores()
        # Compare with the OTHER patients, so edits to this patient rank fairly.
        rank = sum(score > result.score for pid, score in cohort.items() if pid != patient_id) + 1

        score_col, detail_col = st.container(key="iss_wrap_det_risk").columns([1, 2], gap="large")
        with score_col:
            st.metric("Demo deviation score", f"{result.score:g}",
                      help="Sum of the factor points in section 4. Not a probability.")
            risk_indicator("unknown", "Risk category: not assigned",
                           "No validated cut-off exists for this demo score.")
        with detail_col:
            badge_row([(RESULT_TYPE_SIMULATED, "info", False), ("Not a probability", "neutral", False)])
            key_value_grid([
                ("Model status", "Simulated demonstration"),
                ("Latest observation used", f"{result.latest_time:%d %b %Y, %H:%M}" if result.latest_time is not None
                 else "None – no vital-sign readings"),
                ("Inputs available", f"{result.items_used} of {result.items_expected}"),
                ("Rank among synthetic patients", f"{rank} of {len(cohort)} (1 = highest)"),
                ("Factors adding points", f"{sum(c.points > 0 for c in result.contributions)}"),
                ("Data notes", f"{len(result.missing)} missing · {len(result.invalid)} invalid"),
            ])
            st.caption(
                "Higher means the readings sit further outside the illustrative ranges, or moved further "
                "out over the last 24 h. Rank is against the unedited records of the other synthetic patients."
            )


def _render_what_if(vitals, patient_id):
    """Let the user edit the latest readings to see how the score reacts."""
    edited = vitals.copy()
    latest = vitals.iloc[-1]
    with st.expander("Demo: change the latest readings or simulate missing data", icon=":material/tune:"):
        st.caption(
            "Hypothetical edits for exploring how the demo score behaves. Values outside the "
            "data-entry limits are reported as invalid. Edits are not saved and change "
            "nothing in the record."
        )
        cols = st.columns(3)
        for i, key in enumerate(SCORE_VITALS):
            info = VITAL_SIGNS[key]
            step = 0.1 if key == "temperature" else 1.0
            value = cols[i % 3].number_input(
                f"{info['label']} ({info['unit']})",
                value=float(latest[key]),
                step=step,
                format="%.1f" if key == "temperature" else "%.0f",
                key=f"det_edit_{patient_id}_{key}",
            )
            edited.loc[edited.index[-1], key] = value

        missing_latest = st.multiselect(
            "Make the LATEST reading of these vital signs missing",
            options=SCORE_VITALS,
            format_func=lambda k: VITAL_SIGNS[k]["label"],
            key=f"det_missing_{patient_id}",
        )
        for key in missing_latest:
            edited.loc[edited.index[-1], key] = np.nan

        removed = st.multiselect(
            "Remove ALL readings of these vital signs",
            options=SCORE_VITALS,
            format_func=lambda k: VITAL_SIGNS[k]["label"],
            key=f"det_remove_{patient_id}",
        )
        for key in removed:
            edited[key] = np.nan
    if not edited.equals(vitals):
        st.info("The score above uses your **edited** readings (demo only).", icon=":material/edit:")
    return edited


def _render_data_notes(result):
    for note in result.invalid:
        st.warning(f"**Invalid input ignored:** {note}", icon=":material/error:")
    if result.missing:
        st.warning(
            "**Missing data – the score may be lower than it would be with complete data:**\n"
            + "\n".join(f"- {note}" for note in result.missing),
            icon=":material/help:",
        )


# ---------------------------------------------------------------------------
# 2. Vital signs and labs
# ---------------------------------------------------------------------------

def _render_vital_cards(vitals):
    latest = vitals.iloc[-1]
    label(f"Latest recorded vital signs · {latest['timestamp']:%d %b %Y, %H:%M} · mini-trends show the last 24 h")
    render_vital_metrics(vitals)


def _render_labs(patient):
    labs, invalid = lab_inputs(patient)
    with card_container("det_labs"):
        label(f"Laboratory inputs · {len(labs)} of {len(DEMO_LABS)} demo tests recorded")
        if labs:
            fig = lab_range_chart(labs)
            fig.update_layout(title_text="", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch", key="det_lab_chart")
        else:
            st.info("None of the demo lab tests are recorded for this patient.")
        not_recorded = [t for t in DEMO_LABS if t not in labs]
        if not_recorded:
            st.caption("Not recorded for this patient: " + ", ".join(not_recorded) + ".")
        for note in invalid:
            st.warning(note, icon=":material/error:")
        st.caption(
            "Each synthetic lab test has a single result (no history), so labs have no trend. "
            "Shaded band = the test's own reference range from the record."
        )
        if labs:
            with st.expander("Show lab values"):
                table = pd.DataFrame(
                    [
                        {"Test": t, "Value": v, "Unit": u, "Reference range": f"{lo:g}–{hi:g}"}
                        for t, (v, u, lo, hi) in labs.items()
                    ]
                )
                st.dataframe(table, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# 3. Trend analysis
# ---------------------------------------------------------------------------

def _render_trends(vitals):
    """`vitals` are the scored (possibly edited) readings, so the score trend matches section 1."""
    with card_container("det_trend"):
        title_col, window_col = st.container(key="iss_wrap_det_trend_head").columns([3, 2], vertical_alignment="center")
        with title_col:
            label("Observation history")
        with window_col, st.container(horizontal=True, horizontal_alignment="right"):
            window = st.segmented_control(
                "Time window (most recent)", [24, 48, 72], default=72, format_func=lambda h: f"{h} h",
                key="det_trend_window", label_visibility="collapsed",
            ) or 72
        selected = st.segmented_control(
            "Measure", TREND_OPTIONS, default="heart_rate", format_func=TREND_LABELS.get,
            key="det_trend_metric", label_visibility="collapsed",
        ) or "heart_rate"

        cutoff = vitals["timestamp"].max() - pd.Timedelta(hours=window)
        window_df = vitals[vitals["timestamp"] >= cutoff]
        if selected == SCORE_TREND:
            history = score_history(vitals)
            history = history[history["timestamp"] >= cutoff]
            if history.empty:
                st.info("Not enough valid vital-sign readings to show the demo score over time.")
                return
            fig = score_history_chart(history)
            note = (
                "Vital-sign part of the demo score, recalculated at each reading using only data available "
                "at that time. Labs are excluded here because they have no timestamps."
            )
        else:
            fig = blood_pressure_chart(window_df) if selected == "blood_pressure" else vital_trend_chart(window_df, selected)
            note = "Shaded bands show the illustrative adult ranges that the demo score also uses."
        fig.update_layout(height=330, title_text="", margin=dict(t=30))
        st.plotly_chart(fig, width="stretch", key="det_trend_chart")
        st.caption(note)


# ---------------------------------------------------------------------------
# 4. Risk contributors
# ---------------------------------------------------------------------------

def _render_contributors(result):
    with card_container("det_contrib"):
        if not result.contributions:
            st.info("Model explanation not available: there is no demo score to explain.")
            return
        label("Points added by each input · relative contribution to the demo score")
        adding = [c for c in result.contributions if c.points > 0]
        if adding:
            st.plotly_chart(contribution_breakdown_chart(result.contributions), width="stretch",
                            key="det_contrib_chart")
            others = len(result.contributions) - len(adding)
            st.caption(
                f"{others} other available input{'s' if others != 1 else ''} add 0 points (within range, not worsening). "
                "This is the score's own arithmetic breakdown, not a learned feature importance."
            )
        else:
            st.info(
                "All available inputs are within their illustrative ranges, so no factor adds "
                "points. This does **not** mean the patient is safe.",
                icon=":material/info:",
            )

        with st.expander("How each factor was scored"):
            st.dataframe(contributions_table(result), hide_index=True, width="stretch")
            st.caption(
                "Outside range = distance outside the range in multiples of the range width "
                f"(capped at {MAX_PART_POINTS:g}). Worsening = increase in that distance over the "
                f"last {TREND_WINDOW_HOURS} h (vital signs only, capped at {MAX_PART_POINTS:g}). "
                "Points = outside range + worsening. All items are weighted equally."
            )


# ---------------------------------------------------------------------------
# 5. Model information
# ---------------------------------------------------------------------------

def _render_model_info():
    available, reason = labeled_data_check()
    labelled = sum(p.get(OUTCOME_LABEL_KEY) is not None for p in PATIENTS)
    with card_container("det_model", muted=True):
        key_value_grid([
            ("Model type", "No trained model in this module · arithmetic deviation score"),
            ("Training status", "Labelled data found – no model implemented" if available else "Not trained"),
            ("Dataset status", f"{len(PATIENTS)} synthetic patients · {labelled} with outcome labels"),
            ("Evaluation status", "Not evaluated – no trained model" if available
             else "Not evaluated – no outcomes to measure against"),
            ("Prediction mode", RESULT_TYPE_SIMULATED),
            ("Inputs", f"{len(SCORE_VITALS)} vital signs + {len(DEMO_LABS)} demo lab tests"
                       + ("" if USES_MEDICATION_DATA else " · medications not used")),
            ("Weighting", f"Equal · each part capped at {MAX_PART_POINTS:g} point"),
            ("Trend window", f"{TREND_WINDOW_HOURS} h"),
        ])
        st.caption(reason)


# ---------------------------------------------------------------------------
# 6. Limitations and disclaimer
# ---------------------------------------------------------------------------

def _render_limitations():
    with card_container("det_limits", muted=True):
        st.subheader("Limitations and disclaimer")
        info_panel(
            "This page is part of a synthetic college demonstration. It is not a clinically validated "
            "decision-support system, does not predict what will happen to any person and must not be "
            "used for care decisions.",
            title="Synthetic college demonstration",
            tone="neutral",
        )
        st.markdown(
            """
- **Not a prediction model.** No model was trained or validated for this module. The synthetic
  data has no outcome labels, so there is nothing to learn from or to measure accuracy against.
  A separately trained model (on eICU demo data) is available on the ICU Risk Assessment page;
  it does not power this score.
- **Not a probability.** A score of 2 does not mean "twice the risk" or any chance of an event.
- **Arbitrary design choices.** The ranges are illustrative adult ranges, every item is weighted
  equally, and the caps were chosen only so that one item cannot dominate the demo.
- **Missing data lowers the score.** A missing reading adds no points, so incomplete data can
  make a patient look better than they are.
- **No context.** Age, diagnoses, medications, oxygen therapy, and a patient's personal
  normal values are ignored.
- **Real early-warning systems** use validated scores or models tested on real outcomes and
  are always interpreted by clinicians. This demo is not one of them.
"""
        )
