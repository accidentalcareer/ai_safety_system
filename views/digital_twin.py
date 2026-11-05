"""
Patient Digital Twin page.

Shows an editable, simulated copy of the selected synthetic patient and
re-runs the existing Drug Interaction Checker and Deterioration demo score
after each change. All logic lives in utils/digital_twin.py; this file only
draws widgets and calls it.

The page follows the flow PATIENT STATE -> CHANGE -> REASSESSMENT ->
UPDATED SAFETY STATE. Which modules were reassessed is read from the
twin's own event record, so the page never claims a reassessment that the
logic did not run.

Every change is made inside a button callback, so it runs exactly once per
click and is never recorded again on a normal Streamlit rerun.
"""

import numpy as np
import streamlit as st

from data.drug_interaction_rules import INTERACTION_RULES
from data.synthetic_patients import VITAL_SIGNS, classify_value, generate_vitals, get_patient
from ui.components import (
    badge_row,
    card_container,
    fact_rows,
    flow_steps,
    label,
    metric_card_html,
    metric_grid,
    page_header,
    section_header,
    status_grid,
    tile_grid,
    tile_html,
    timeline,
    timeline_entry_html,
)
from utils.charts import contributions_chart
from utils.deterioration import (
    DEMO_LABS,
    RESULT_TYPE_SIMULATED,
    SCORE_VITALS,
    contributions_table,
)
from utils.digital_twin import (
    MODULE_DETERIORATION,
    MODULE_INTERACTIONS,
    STATUS_NEEDS_REASSESSMENT,
    TwinInputError,
    TwinRegistry,
    format_vital,
    timeline_table,
)
from utils.drug_interactions import KNOWN_DRUGS, RESULT_TYPE_RULE, display_name, is_known_drug, normalize_name
from views.drug_interactions import render_interaction_card

REGISTRY_KEY = "twin_registry"
FEEDBACK_KEY = "twin_feedback"
RULE_BADGE = RESULT_TYPE_RULE
SIM_BADGE = RESULT_TYPE_SIMULATED
MODULES = [MODULE_INTERACTIONS, MODULE_DETERIORATION]
MODULE_SHORT = {MODULE_INTERACTIONS: "Interaction checker", MODULE_DETERIORATION: "Demo score"}

# Events that are not a simulated change of the patient's state.
NON_CHANGE_EVENTS = ("Twin created", "Reset", "Manual reassessment")
# Vital cards in the physiological panel; blood pressure combines systolic/diastolic.
STATE_VITALS = ["heart_rate", "blood_pressure", "spo2", "resp_rate", "temperature"]
STATUS_SHORT = {"Low": "Below range", "High": "Above range", "Normal": "In range"}


def get_registry():
    """One TwinRegistry per browser session (one twin per patient inside it)."""
    if REGISTRY_KEY not in st.session_state:
        st.session_state[REGISTRY_KEY] = TwinRegistry()
    return st.session_state[REGISTRY_KEY]


def _key(patient_id, name):
    """Widget keys are per patient so patients never share inputs."""
    return f"twin_{patient_id}_{name}"


# ---------------------------------------------------------------------------
# Button callbacks (run once per click, before the page is redrawn)
# ---------------------------------------------------------------------------

def _apply(patient_id, action):
    """Run one twin action and store a message to show after the rerun."""
    twin = get_registry().get(patient_id)
    try:
        event = action(twin)
    except TwinInputError as err:
        st.session_state[FEEDBACK_KEY] = (patient_id, "error", str(err))
        return False
    st.session_state[FEEDBACK_KEY] = (
        patient_id, "success", f"Recorded event #{event.event_id}: {event.change_type} – {event.item}.",
    )
    return True


def _on_add_medication(patient_id):
    typed = st.session_state.get(_key(patient_id, "med_typed"), "") or ""
    name = typed.strip() or st.session_state.get(_key(patient_id, "med_choice"))
    if _apply(patient_id, lambda twin: twin.add_medication(name)):
        st.session_state[_key(patient_id, "med_typed")] = ""


def _on_remove_medication(patient_id):
    name = st.session_state.get(_key(patient_id, "med_remove"))
    _apply(patient_id, lambda twin: twin.remove_medication(name))


def _on_update_vital(patient_id):
    key = st.session_state.get(_key(patient_id, "vital_key"))
    if st.session_state.get(_key(patient_id, "vital_missing")):
        value = None
    else:
        value = st.session_state.get(_key(patient_id, f"vital_value_{key}"))
        if value is None:
            st.session_state[FEEDBACK_KEY] = (patient_id, "error", "Enter a value, or tick 'mark as missing'.")
            return
    _apply(patient_id, lambda twin: twin.update_vital(key, value))


def _on_update_lab(patient_id):
    test = st.session_state.get(_key(patient_id, "lab_test"))
    value = st.session_state.get(_key(patient_id, f"lab_value_{test}"))
    _apply(patient_id, lambda twin: twin.update_lab(test, value))


def _on_reassess(patient_id):
    _apply(patient_id, lambda twin: twin.reassess())


def _on_reset(patient_id):
    if _apply(patient_id, lambda twin: twin.reset()):
        # Clear this patient's edit widgets so they show the original values again.
        prefix = _key(patient_id, "")
        for name in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefix)]:
            del st.session_state[name]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render(patient_id):
    """Draw the Patient Digital Twin page for one patient."""
    patient = get_patient(patient_id)
    twin = get_registry().get(patient_id)

    page_header(
        "Patient Digital Twin",
        subtitle=(
            f"Simulate patient-state changes and observe system reassessment. {patient['display_name']} · "
            f"Patient ID **{patient_id}** · Simulated copy of a synthetic record."
        ),
        eyebrow="Simulation",
        badges=[("Simulation", "info"), ("Synthetic patient", "neutral"),
                (f"{_changes_since_reset(twin)} simulated change(s) active", "neutral", False)],
    )
    with st.container(key="iss_notice_twin"):
        st.warning(
            "**Simulation only – not clinical decision support.** Changes here edit a simulated copy "
            "of a fictional patient. They do not recommend, prescribe or perform any real medication "
            "change, and the other pages keep showing the original synthetic record.",
            icon=":material/science:",
        )
    flow_steps([
        ("Step 1", "Patient state", "The twin's current simulated values.", None, None),
        ("Step 2", "Change", "One explicit, applied edit.", None, None),
        ("Step 3", "Reassessment", "Only the modules that use the changed data.", None, None),
        ("Step 4", "Updated safety state", "New module outputs, logged on the timeline.", None, None),
    ])

    section_header("Current patient state", "The simulated twin, compared with the original synthetic record.", step=1)
    _render_overview(twin)
    physio_col, meds_col = st.container(key="iss_wrap_twin_state").columns([3, 2], gap="medium")
    with physio_col:
        _render_physiology(twin)
    with meds_col:
        _render_medication_exposure(twin)

    section_header(
        "Simulate a change",
        "Pick a change type, fill in the fields and press Apply. Nothing changes until you do.",
        step=2,
    )
    _render_controls(twin)

    section_header("System response", "What the most recent action changed and which modules ran.", step=3)
    _render_response(twin)

    section_header("Event timeline", "Every action on this patient's twin, newest first.", step=4)
    _render_timeline(twin)

    section_header("System status", step=5)
    _render_system_status(twin)


def _changes_since_reset(twin):
    """Simulated state changes since the twin was created or last reset."""
    changes = 0
    for event in reversed(twin.events):
        if event.change_type in ("Twin created", "Reset"):
            break
        changes += event.change_type not in NON_CHANGE_EVENTS
    return changes


# ---------------------------------------------------------------------------
# 1. Current patient state
# ---------------------------------------------------------------------------

def _render_overview(twin):
    n_rules = len(twin.interactions.interactions)
    # Fixed-width cards in a wrapping row: four across on desktop, two on a tablet.
    row = st.container(key="twin_overview", horizontal=True, horizontal_alignment="distribute", gap="small")
    row.metric("Simulated medications", len(twin.medication_names), border=True, width=200)
    row.metric("Documented rule matches", n_rules, border=True, help=RULE_BADGE, width=200)
    row.metric(
        "Demo deviation score (simulated)", f"{twin.deterioration.score:g}", border=True,
        help=f"{SIM_BADGE}. Sum of points, not a probability.", width=200,
    )
    row.metric("Timeline events", len(twin.events), border=True, width=200)


def _render_physiology(twin):
    original = generate_vitals(twin.patient_id).iloc[-1]
    latest = twin.latest_vitals()
    with card_container("twin_physio", fill_height=True):
        label(f"Physiological state · latest reading {latest['timestamp']:%d %b %Y, %H:%M} (synthetic clock)")
        cards = []
        for key in STATE_VITALS:
            series = ["systolic_bp", "diastolic_bp"] if key == "blood_pressure" else [key]
            now = [latest[s] for s in series]
            was = [original[s] for s in series]
            changed = [format_vital(s, a) for s, a in zip(series, now)] != [format_vital(s, b) for s, b in zip(series, was)]
            info = VITAL_SIGNS[series[0]]
            if any(np.isnan(v) for v in now):
                value, status, tone = "—", "Missing", "warning"
            else:
                value = "/".join(f"{v:g}" for v in now)
                status = STATUS_SHORT[classify_value(now[0], info["low"], info["high"])]
                tone = "neutral" if status == "In range" else "warning"
            name = "Blood pressure" if key == "blood_pressure" else info["short"]
            was_text = "/".join("missing" if np.isnan(v) else f"{v:g}" for v in was)
            cards.append(metric_card_html(
                name, value, unit=info["unit"], status=status, status_tone=tone,
                caption=f"Changed · was {was_text}" if changed else "Original value",
            ))
        metric_grid(cards)

        label("Laboratory values")
        original_values = {test: value for test, value, *_ in get_patient(twin.patient_id)["labs"]}
        rows = []
        for test, value, unit, low, high in twin.labs:
            notes = [f"ref {low:g}–{high:g}", STATUS_SHORT[classify_value(value, low, high)].lower()]
            if test in DEMO_LABS:
                notes.append("score input")
            if original_values.get(test) != value:
                notes.append(f"changed · was {original_values.get(test):g}")
            rows.append((test, f"{value:g} {unit}", " · ".join(notes), False))
        if rows:
            fact_rows(rows)
        else:
            st.info("No lab values are recorded for this patient.")
        st.caption("Status is a plain range comparison, not a clinical interpretation.")


def _render_medication_exposure(twin):
    result = twin.interactions
    involved = {}
    for interaction in result.interactions:
        for drug in (interaction.drug_a, interaction.drug_b):
            involved[drug] = involved.get(drug, 0) + 1

    with card_container("twin_meds", fill_height=True):
        label("Medication exposure")
        n = len(result.interactions)
        badge_row([
            (f"{n} documented interaction{'s' if n != 1 else ''}", "warning" if n else "neutral"),
            (f"{len(result.unknown_drugs)} not checkable", "warning" if result.unknown_drugs else "neutral", False),
        ])
        tiles = []
        for med in twin.medications:
            name = med.get("name")
            if not name:
                continue
            key = normalize_name(name)
            badges, tone = [], None
            if not is_known_drug(name):
                badges, tone = [("Not in sample dataset", "warning")], "warning"
            elif key in involved:
                count = involved[key]
                badges, tone = [(f"In {count} documented interaction{'s' if count != 1 else ''}", "warning")], "warning"
            else:
                badges = [("No rule matched", "neutral")]
            if med.get("indication") == "Simulated demo addition":
                badges.append(("Simulated addition", "info", False))
            tiles.append(tile_html(
                name, subtitle=" · ".join(str(v) for v in (med.get("dose"), med.get("frequency")) if v),
                badges=badges, tone=tone,
            ))
        if tiles:
            tile_grid(tiles)
        else:
            st.info("The simulated medication list is empty.")
        st.caption(
            f"Checked against {len(INTERACTION_RULES)} sample rules (names only). No match does not mean "
            "the combination is safe."
        )


# ---------------------------------------------------------------------------
# 2. Simulation controls
# ---------------------------------------------------------------------------

def _render_controls(twin):
    pid = twin.patient_id
    with card_container("twin_controls", tone="info"):
        add_tab, remove_tab, vital_tab, lab_tab = st.tabs(
            ["Add medication", "Remove medication", "Change vital sign", "Change lab value"]
        )
        with add_tab:
            _render_add_medication(twin)
        with remove_tab:
            _render_remove_medication(twin)
        with vital_tab:
            _render_vital_control(twin)
        with lab_tab:
            _render_lab_control(twin)

        _render_feedback(pid)
        st.divider()
        note_col, buttons_col = st.columns([1, 1], vertical_alignment="center")
        note_col.caption(
            "Hypothetical edits to a simulated copy, used to show how the modules react. Not a "
            "prescribing suggestion. Reset restores the original sample values; the timeline is kept."
        )
        with buttons_col, st.container(horizontal=True, horizontal_alignment="right"):
            st.button("Reassess all modules", on_click=_on_reassess, args=(pid,),
                      key=_key(pid, "btn_reassess"), icon=":material/refresh:")
            st.button("Reset patient state", on_click=_on_reset, args=(pid,),
                      key=_key(pid, "btn_reset"), icon=":material/restart_alt:")


def _render_add_medication(twin):
    pid = twin.patient_id
    on_list = {normalize_name(name) for name in twin.medication_names}
    choices = sorted(display_name(d) for d in KNOWN_DRUGS if d not in on_list)
    pick_col, type_col = st.columns(2)
    with pick_col:
        if choices:
            st.selectbox("Medication from the sample dataset", choices, key=_key(pid, "med_choice"))
        else:
            st.caption("Every sample-dataset medication is already on the list.")
    type_col.text_input(
        "…or type any name (overrides the list)", key=_key(pid, "med_typed"),
        placeholder="e.g. Albuterol or Unknownium",
    )
    st.caption("Reassesses: Drug Interaction Checker. The deterioration score has no medication inputs.")
    st.button("Apply change", on_click=_on_add_medication, args=(pid,), key=_key(pid, "btn_add"),
              type="primary", icon=":material/add:")


def _render_remove_medication(twin):
    pid = twin.patient_id
    if not twin.medication_names:
        st.info("No medications left to remove.")
        return
    st.selectbox("Medication to remove", twin.medication_names, key=_key(pid, "med_remove"))
    st.caption("Reassesses: Drug Interaction Checker. The deterioration score has no medication inputs.")
    st.button("Apply change", on_click=_on_remove_medication, args=(pid,), key=_key(pid, "btn_remove"),
              type="primary", icon=":material/remove:")


def _render_vital_control(twin):
    pid = twin.patient_id
    latest = twin.latest_vitals()
    key_col, value_col = st.columns(2)
    with key_col:
        key = st.selectbox(
            "Vital sign", SCORE_VITALS, format_func=lambda k: VITAL_SIGNS[k]["label"],
            key=_key(pid, "vital_key"),
        )
    current = latest[key]
    is_temp = key == "temperature"
    with value_col:
        st.number_input(
            f"New value ({VITAL_SIGNS[key]['unit']}) – currently {format_vital(key, current)}",
            value=None if np.isnan(current) else float(current),
            step=0.1 if is_temp else 1.0,
            format="%.1f" if is_temp else "%.0f",
            key=_key(pid, f"vital_value_{key}"),
        )
    st.checkbox("Mark the latest reading as missing instead", key=_key(pid, "vital_missing"))
    st.caption(
        "Replaces the latest synthetic reading. Values outside the deterioration module's data-entry "
        "limits are rejected. Reassesses: Deterioration demo score (and the interaction context if a "
        "matched rule lists this value)."
    )
    st.button("Apply change", on_click=_on_update_vital, args=(pid,), key=_key(pid, "btn_vital"),
              type="primary", icon=":material/check:")


def _render_lab_control(twin):
    pid = twin.patient_id
    tests = [entry[0] for entry in twin.labs]
    if not tests:
        st.info("No lab values are recorded for this patient.")
        return
    test_col, value_col = st.columns(2)
    with test_col:
        test = st.selectbox("Lab test", tests, key=_key(pid, "lab_test"))
    unit = next(entry[2] for entry in twin.labs if entry[0] == test)
    with value_col:
        st.number_input(
            f"New value ({unit}) – currently {twin.lab_value(test):g}",
            value=float(twin.lab_value(test)), step=0.1, format="%.2f",
            key=_key(pid, f"lab_value_{test}"),
        )
    used = "is" if test in DEMO_LABS else "is **not**"
    st.caption(f"{test} {used} one of the lab values used by the demo score.")
    st.button("Apply change", on_click=_on_update_lab, args=(pid,), key=_key(pid, "btn_lab"),
              type="primary", icon=":material/check:")


def _render_feedback(patient_id):
    feedback = st.session_state.pop(FEEDBACK_KEY, None)
    if not feedback or feedback[0] != patient_id:
        return
    _, level, message = feedback
    if level == "error":
        st.error(message, icon=":material/error:")
    else:
        st.success(message, icon=":material/check_circle:")


# ---------------------------------------------------------------------------
# 3. System response
# ---------------------------------------------------------------------------

def _module_reasons(event, module):
    """The event's own reason lines for one module (text after 'Module: ')."""
    prefix = f"{module}: "
    return [reason[len(prefix):] for reason in event.reasons if reason.startswith(prefix)]


def _render_response(twin):
    event = twin.events[-1]
    if event.change_type == "Twin created":
        st.info(
            "No simulated change yet. The twin was created from the original synthetic record and both "
            "modules assessed it. Apply a change above to see the reassessment here.",
            icon=":material/info:",
        )
    else:
        module_lines = []
        for module in MODULES:
            ran = module in event.modules_reassessed
            module_lines.append((f"{module}: {'reassessed' if ran else 'not reassessed'}",
                                 "info" if ran else "neutral", ran))
        general = [r for r in event.reasons if not any(r.startswith(f"{m}: ") for m in MODULES)]
        detail = " ".join(
            general + [f"{MODULE_SHORT[m]} – {' '.join(_module_reasons(event, m))}" for m in MODULES
                       if _module_reasons(event, m)]
        )
        flow_steps([
            ("Previous state", "Medication list" if event.change_type.startswith("Medication") else event.item,
             event.previous_value, None, "neutral"),
            ("Change detected", event.change_type, f"#{event.event_id} · {event.timestamp:%H:%M:%S} · now {event.new_value}",
             None, "info"),
            ("Modules reassessed", f"{len(event.modules_reassessed)} of {len(MODULES)} modules ran", detail,
             module_lines, "info"),
            ("Current state", "Updated safety state", event.status, [(s, "neutral", False) for s in event.sources],
             "warning" if twin.deterioration_status == STATUS_NEEDS_REASSESSMENT else "info"),
        ], prominent=True)

    if twin.deterioration_status == STATUS_NEEDS_REASSESSMENT:
        st.warning(f"**Deterioration score needs reassessment.** {twin.deterioration_note}",
                   icon=":material/sync_problem:")

    interactions_col, score_col = st.container(key="iss_wrap_twin_response").columns(2, gap="medium")
    with interactions_col, card_container("twin_out_interactions", muted=True):
        label(f"Drug Interaction Checker · {RULE_BADGE}")
        _render_interactions(twin)
    with score_col, card_container("twin_out_deterioration", muted=True):
        label(f"Deterioration · {SIM_BADGE}")
        _render_deterioration(twin)


def _render_interactions(twin):
    result = twin.interactions
    if result.unknown_drugs:
        st.warning(
            "**Not in the sample dataset – could not be checked:** " + ", ".join(result.unknown_drugs)
            + ". The checker cannot say anything about these names.",
            icon=":material/help:",
        )
    if not result.interactions:
        st.info(
            "No interaction rules matched in this small sample dataset. **This does not mean the "
            f"combination is safe** – the sample covers only {len(INTERACTION_RULES)} documented pairs.",
            icon=":material/info:",
        )
        return
    # Context values come from the twin, not the original record. Missing
    # readings are dropped so they show as "Not recorded".
    labs = twin.lab_table()
    latest = twin.latest_vitals().dropna()
    for interaction in result.interactions:
        render_interaction_card(interaction, labs, latest)


def _render_deterioration(twin):
    result = twin.deterioration
    st.markdown(
        f"**Status:** {twin.deterioration_status} · **Demo deviation score:** {result.score:g} · "
        f"**Inputs available:** {result.items_used} of {result.items_expected}"
    )
    st.caption(
        "Not a probability or a clinical prediction, and no risk band is shown because no "
        "validated cut-off exists. This score does not use medication data."
    )
    for note in result.invalid:
        st.warning(f"**Invalid input ignored:** {note}", icon=":material/error:")
    if result.missing:
        with st.expander(f"Missing-data notes ({len(result.missing)})"):
            st.markdown("\n".join(f"- {note}" for note in result.missing))
    if not result.contributions:
        st.info("Not enough valid data to calculate a demo score.")
        return
    if any(c.points > 0 for c in result.contributions):
        fig = contributions_chart(result.contributions)
        fig.update_layout(title_text="", margin=dict(t=10))
        # Headroom so the value labels outside the bars are not clipped.
        fig.update_xaxes(range=[0, max(c.points for c in result.contributions) * 1.25])
        st.plotly_chart(fig, width="stretch", key=f"twin_contrib_{twin.patient_id}")
    with st.expander("How each factor was scored"):
        st.dataframe(contributions_table(result), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# 4. Event timeline
# ---------------------------------------------------------------------------

def _render_timeline(twin):
    entries = []
    for event in reversed(twin.events):
        modules = [(m, "info", False) for m in event.modules_reassessed] or [("No module reassessed", "neutral", False)]
        entries.append(timeline_entry_html(
            f"#{event.event_id} · {event.change_type} · {event.item}",
            time=f"{event.timestamp:%d %b %Y, %H:%M:%S}",
            body=f"{event.previous_value} → {event.new_value}",
            meta=f"Result: {event.status}",
            tone="neutral" if event.change_type in NON_CHANGE_EVENTS else "info",
            badges=modules,
        ))
    with card_container("twin_timeline", height=460 if len(entries) > 4 else None):
        timeline(entries)
    st.caption("Time = when the action was made in this session (wall clock). Only this patient's events are shown.")
    with st.expander("Show as table, with the reason for each module decision"):
        st.dataframe(
            timeline_table(twin.events),
            hide_index=True,
            width="stretch",
            column_config={
                "Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                "Why": st.column_config.TextColumn(width="large"),
            },
        )


# ---------------------------------------------------------------------------
# 5. System status
# ---------------------------------------------------------------------------

def _render_system_status(twin):
    changes = _changes_since_reset(twin)
    result = twin.interactions
    n = len(result.interactions)
    pending = twin.deterioration_status == STATUS_NEEDS_REASSESSMENT
    last = twin.events[-1]
    status_grid([
        ("Patient state",
         f"Modified · {changes} change{'s' if changes != 1 else ''} since reset" if changes else "Original sample values",
         "Simulated copy; the other pages show the original record.", "info" if changes else "neutral"),
        ("Drug interaction engine",
         f"Current · {n} documented rule match{'es' if n != 1 else ''}",
         f"{RULE_BADGE} · {len(INTERACTION_RULES)} sample rules"
         + (f" · {len(result.unknown_drugs)} name(s) not checkable" if result.unknown_drugs else ""), "info"),
        ("Deterioration engine",
         f"{twin.deterioration_status} · score {twin.deterioration.score:g}",
         f"{SIM_BADGE} · no medication inputs", "warning" if pending else "info"),
        ("Integration engine",
         f"{len(twin.events)} event{'s' if len(twin.events) != 1 else ''} logged",
         f"Last: {last.change_type} at {last.timestamp:%H:%M:%S}", "neutral"),
    ])
    _render_source_legend()


def _render_source_legend():
    with st.expander("Where do the outputs come from?"):
        st.markdown(
            f"- **{RULE_BADGE}** – Drug Interaction Checker: fixed rules copied from official "
            "product labelling. Rules match on drug names only.\n"
            f"- **{SIM_BADGE}** – Deterioration module: transparent arithmetic on vital signs and "
            "a few lab values. Not a probability and not a prediction.\n"
            "- **Simulated logic (demo state change)** – the edit you made to the simulated copy.\n"
            "- **Trained model** – *not used by the Digital Twin.* A separately trained model is "
            "available on the ICU Risk Assessment page."
        )
