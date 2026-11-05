"""
ICU Risk Assessment page – model-derived ICU risk score.

Collects readable clinical inputs and passes them unchanged to the ML layer:
    ml/feature_adapter.py  (validation, 24h aggregation, 126 model features)
    ml/inference.py        (saved pipeline, risk score, demonstration category)
This page contains no model, validation or aggregation logic of its own; it only
parses the typed text into numbers and displays the result.

The score is an UNCALIBRATED model risk score from a proof-of-concept Logistic
Regression model trained on the eICU demo dataset. It is never shown as a
probability or a percentage, and the categories are model categories, not
clinical ones. The Deterioration module does not use this model.
"""

import re

import streamlit as st

from ml import feature_adapter
from ui.components import (
    card_container,
    key_value_grid,
    label,
    page_header,
    section_header,
    summary_block,
)

# The adapter's own `inference` module (same objects, so its error classes match).
inference = feature_adapter.inference

DISCLAIMER = ("Proof-of-concept model risk score. Not a clinical diagnosis or validated "
              "clinical decision-support tool.")
NOT_RECORDED = "Not recorded"

# Full names, used in validation messages and the completeness notes.
LABELS = {
    "age": "Age", "gender": "Gender", "unittype": "Unit type", "unitstaytype": "ICU stay type",
    "admissionweight": "Admission weight", "admissionheight": "Admission height",
    "heartrate": "Heart rate", "sao2": "SpO₂", "respiration": "Respiratory rate",
    "temperature": "Temperature", "systemicsystolic": "Systemic systolic BP",
    "systemicdiastolic": "Systemic diastolic BP", "systemicmean": "Systemic mean BP",
    "noninvasivesystolic": "Non-invasive systolic BP", "noninvasivediastolic": "Non-invasive diastolic BP",
    "noninvasivemean": "Non-invasive mean BP", "glucose": "Glucose", "sodium": "Sodium",
    "potassium": "Potassium", "chloride": "Chloride", "bicarbonate": "Bicarbonate",
    "creatinine": "Creatinine", "bun": "BUN", "wbc": "WBC", "hemoglobin": "Hemoglobin",
    "platelets": "Platelets", "calcium": "Calcium", "magnesium": "Magnesium", "lactate": "Lactate",
    "intubated": "Intubated", "vent": "Ventilated", "dialysis": "Dialysis",
}
# Shorter widget labels inside the blood-pressure boxes, which already name the method.
FIELD_LABELS = {
    "noninvasivesystolic": "Systolic", "noninvasivediastolic": "Diastolic", "noninvasivemean": "Mean / MAP",
    "systemicsystolic": "Systolic", "systemicdiastolic": "Diastolic", "systemicmean": "Mean / MAP",
}
STAY_TYPE_LABELS = {"admit": "Admission", "readmit": "Readmission", "transfer": "Transfer"}
FLAG_OPTIONS = ["No", "Yes"]

BEDSIDE = ["heartrate", "sao2", "respiration", "temperature"]
BP_GROUPS = [
    ("Cuff · non-invasive", ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"]),
    ("Arterial line · invasive", ["systemicsystolic", "systemicdiastolic", "systemicmean"]),
]
LAB_GROUPS = [
    ("Chemistry", ["glucose", "sodium", "potassium", "chloride", "bicarbonate",
                   "creatinine", "bun", "calcium", "magnesium"]),
    ("Hematology", ["wbc", "hemoglobin", "platelets"]),
    ("Other", ["lactate"]),
]
VITALS = BEDSIDE + [f for _, fields in BP_GROUPS for f in fields]
LABS = [f for _, fields in LAB_GROUPS for f in fields]
MEASUREMENTS = VITALS + LABS
FIELDS_PER_ROW = 3

# Measurements recorded for at least 90% of the training ICU stays (computed from the
# `_missing` flags in ml/data/processed/mortality_training.csv). Lacking one of these
# makes a record unusual compared with the data the model learned from.
ROUTINE_MEASUREMENTS = [
    "heartrate", "sao2", "respiration", "noninvasivesystolic", "noninvasivediastolic",
    "noninvasivemean", "glucose", "sodium", "potassium", "chloride", "creatinine", "bun",
    "wbc", "hemoglobin", "platelets", "calcium",
]

# Illustrative values for the "Load example" button: not a real patient.
EXAMPLE = {
    "age": 67.0, "gender": "Male", "unittype": "MICU", "unitstaytype": "admit",
    "admissionweight": 82.0, "admissionheight": 175.0,
    "heartrate": "92, 104, 110, 98", "sao2": "96, 93, 95", "respiration": "18, 24, 22",
    "temperature": "37.4, 38.2", "noninvasivesystolic": "118, 102, 96",
    "noninvasivediastolic": "64, 58, 55", "noninvasivemean": "82, 73, 69",
    "glucose": "148, 176", "sodium": "137, 135", "potassium": "4.2, 3.8", "chloride": "102, 100",
    "bicarbonate": "21, 19", "creatinine": "1.3, 1.7", "bun": "26, 31", "wbc": "12.4, 14.1",
    "hemoglobin": "11.8, 10.9", "platelets": "214, 190", "calcium": "8.6, 8.3", "lactate": "2.6",
    "intubated": "No", "vent": "Yes", "dialysis": "No",
}
NUMBER_FIELDS = ["age", "admissionweight", "admissionheight"]
SELECT_FIELDS = ["gender", "unittype", "unitstaytype"]
FLAG_FIELDS = list(feature_adapter.APACHE_FLAGS)
RESULT_KEY = "icu_result"


def _key(field):
    return f"icu_{field}"


def _unit(field):
    return feature_adapter.UNITS.get(field, "")


def model_status():
    """(text, tone) for the sidebar status: the trained model if it loads, otherwise unavailable."""
    try:
        _, meta = inference.load_model()
    except Exception:  # noqa: BLE001 - any load failure means the model cannot be used
        return "Unavailable", "warning"
    return f"Trained · {meta['model_name']} v{meta['model_version']}", "info"


def render(patient_id=None):
    """Draw the ICU Risk Assessment page. Independent of the synthetic patient selector."""
    status_text, _ = model_status()
    _, _, model_text = status_text.partition(" · ")
    page_header(
        "ICU Risk Assessment",
        subtitle="Model-derived ICU mortality risk score from first-24-hour clinical measurements.",
        eyebrow="Patient safety",
        badges=[("Trained model", "info"),
                (f"{model_text or 'Model unavailable'} · eICU demo data", "neutral"),
                ("Uncalibrated risk score", "neutral")],
    )
    with st.container(key="iss_notice_icu"):
        st.warning(f"**For academic demonstration only.** {DISCLAIMER}", icon=":material/science:")

    head_col, actions_col = st.columns([3, 2], vertical_alignment="bottom")
    with head_col:
        section_header("Patient profile",
                       "Required: Unit type, ICU stay type, intubation, ventilation and dialysis.", step="01")
    with actions_col:
        _render_toolbar()
    _render_profile()

    section_header("Clinical status", "First-day status. All three are required.", step="02")
    _render_status()

    section_header(
        "Vital signs",
        "First 24 hours of the ICU stay. Enter one reading or several readings separated by commas "
        "(e.g. 88, 102, 95); leave a field blank if it was not recorded.",
        step="03",
    )
    _render_vitals()

    section_header(
        "Laboratory results",
        "Same format as vital signs: results from 6 hours before to 24 hours after ICU admission, "
        "in time order.",
        step="04",
    )
    _render_labs()

    section_header("Assessment", "Check the data entered, then run the model.", step="05")
    errors_slot = st.container()  # validation summary, directly above the button
    if _render_action():
        st.session_state[RESULT_KEY] = {**_assess(), "inputs": _snapshot()}
    outcome = st.session_state.get(RESULT_KEY)
    stale = outcome is not None and outcome.get("inputs") != _snapshot()
    if outcome and "errors" in outcome:
        with errors_slot:
            _render_errors(outcome)
    _render_result(outcome, stale)
    _render_model_info()


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

def _clear_form():
    for field in feature_adapter.CLINICAL_FIELDS:
        st.session_state[_key(field)] = "" if field in MEASUREMENTS else None
    st.session_state.pop(RESULT_KEY, None)


def _load_example():
    _clear_form()
    for field, value in EXAMPLE.items():
        st.session_state[_key(field)] = value


def _snapshot():
    """The current form values, to tell whether a shown result still matches the inputs."""
    values = (st.session_state.get(_key(f)) for f in feature_adapter.CLINICAL_FIELDS)
    return tuple(None if v == "" else v for v in values)  # an empty text box counts as blank


def _render_toolbar():
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        st.button("Load example", icon=":material/description:", key="icu_load_example",
                  on_click=_load_example, type="tertiary",
                  help="Fills the form with illustrative values (not a real patient).")
        st.button("Clear form", icon=":material/restart_alt:", key="icu_clear",
                  on_click=_clear_form, type="tertiary")
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.caption("Example: illustrative values only — not a real patient.")


def _row(n=FIELDS_PER_ROW):
    """One row of input columns; fields are placed left to right, so stacking keeps their order."""
    return st.columns(n, vertical_alignment="bottom")


def _number(field, step, fmt, **kwargs):
    st.number_input(f"{LABELS[field]} ({_unit(field)})", value=None, step=step, format=fmt,
                    key=_key(field), placeholder=NOT_RECORDED, **kwargs)


def _render_profile():
    with card_container("icu_profile"):
        # Two per row: selectboxes and steppers stay readable at tablet width. Required fields first.
        unit_col, stay_col = _row(2)
        with unit_col:
            st.selectbox(f"{LABELS['unittype']} *", feature_adapter.CATEGORICAL["unittype"], index=None,
                         key=_key("unittype"), placeholder="Select unit type")
        with stay_col:
            st.selectbox(f"{LABELS['unitstaytype']} *", feature_adapter.CATEGORICAL["unitstaytype"],
                         index=None, format_func=lambda v: STAY_TYPE_LABELS.get(v, v),
                         key=_key("unitstaytype"), placeholder="Select stay type")
        gender_col, age_col = _row(2)
        with gender_col:
            st.selectbox(LABELS["gender"], feature_adapter.CATEGORICAL["gender"], index=None,
                         key=_key("gender"), placeholder=NOT_RECORDED)
        with age_col:
            _number("age", 1.0, "%.0f", help="Ages above 89 are encoded as 90, as in the training data.")
        weight_col, height_col = _row(2)
        with weight_col:
            _number("admissionweight", 0.5, "%.1f")
        with height_col:
            _number("admissionheight", 1.0, "%.0f")


def _render_status():
    with card_container("icu_status"):
        for col, field in zip(_row(), FLAG_FIELDS):
            with col:
                st.segmented_control(f"{LABELS[field]} *", FLAG_OPTIONS, key=_key(field), width="stretch")
        st.caption("APACHE first-day flags, as recorded in the training data.")


def _measurement_rows(fields):
    for start in range(0, len(fields), FIELDS_PER_ROW):
        for col, field in zip(_row(), fields[start:start + FIELDS_PER_ROW]):
            with col:
                st.text_input(f"{FIELD_LABELS.get(field, LABELS[field])} ({_unit(field)})",
                              key=_key(field), placeholder=NOT_RECORDED)


def _render_vitals():
    with card_container("icu_vitals"):
        label("Bedside monitoring")
        _measurement_rows(BEDSIDE)
        st.divider()
        label("Blood pressure")
        for i, (method, fields) in enumerate(BP_GROUPS):
            with card_container(f"icu_bp_{i}"):  # nested, so drawn flat inside the vitals card
                st.caption(f"**{method}**")
                _measurement_rows(fields)


def _render_labs():
    with card_container("icu_labs"):
        for i, (group, fields) in enumerate(LAB_GROUPS):
            if i:
                st.divider()
            label(group)
            _measurement_rows(fields)


def _entered_count():
    return sum(bool((st.session_state.get(_key(f)) or "").strip()) for f in MEASUREMENTS)


def _render_action():
    """Completeness of the form plus the primary button. Returns True when it is pressed."""
    entered = _entered_count()
    with card_container("icu_action", tone="info"):
        info_col, button_col = st.columns(2, gap="large", vertical_alignment="center")
        with info_col:
            label("Data completeness")
            st.markdown(f"**{entered} / {len(MEASUREMENTS)} measurements entered**")
            st.progress(entered / len(MEASUREMENTS))
            st.caption("Unrecorded measurements are handled by the trained preprocessing pipeline; "
                       "limited observations may affect the model output.")
        with button_col:
            clicked = st.button("Assess ICU Risk", type="primary", icon=":material/analytics:",
                                key="icu_submit", width="stretch")
            st.caption("Generates an uncalibrated model risk score from the entered first-24-hour data.")
    return clicked


# ---------------------------------------------------------------------------
# Collect -> feature_adapter -> inference
# ---------------------------------------------------------------------------

def _parse_readings(field, text):
    """'88, 102 95' -> [88.0, 102.0, 95.0]; blank -> None. Returns (readings, error)."""
    tokens = [t for t in re.split(r"[,;\s]+", (text or "").strip()) if t]
    if not tokens:
        return None, None
    readings = []
    for token in tokens:
        try:
            readings.append(float(token))
        except ValueError:
            return None, f"{LABELS[field]}: '{token}' is not a number."
    return readings, None


def _friendly(message):
    """Replace internal field names in an adapter error with their form labels."""
    for field, text in sorted(LABELS.items(), key=lambda kv: -len(kv[0])):
        message = re.sub(rf"(?<![\w']){field}(?![\w'])", text, message)
        message = message.replace(f"'{field}'", text)
    return message


def _messages(errors):
    """Adapter error strings -> one short, readable line per problem (wording only)."""
    lines = []
    for message in errors:
        required = re.match(r"required field\(s\) missing: \[(.*)\]$", message)
        out_of_range = re.match(r"^(\w+)=(\S+) (.*?) is outside the plausible range \[(.+?), (.+?)\](.*)$", message)
        if required:
            lines += [f"{LABELS.get(f, f)} is required." for f in re.findall(r"'([^']+)'", required.group(1))]
        elif out_of_range:
            field, value, unit, low, high, hint = out_of_range.groups()
            line = (f"{LABELS.get(field, field)}: {float(value):g} {unit} is outside the allowed range "
                    f"({low}–{high} {unit}).")
            lines.append(line + (" Enter temperature in °C." if "°C" in hint else ""))
        else:
            lines.append(_friendly(message))
    return lines


def _collect():
    """Form values -> clinical input dict for feature_adapter, parse errors and reading counts."""
    state = st.session_state
    clinical, errors, counts = {}, [], {}
    for field in NUMBER_FIELDS + SELECT_FIELDS:
        clinical[field] = state.get(_key(field))
    for field in FLAG_FIELDS:
        choice = state.get(_key(field))
        clinical[field] = None if choice is None else choice == "Yes"
    for field in MEASUREMENTS:
        readings, error = _parse_readings(field, state.get(_key(field)))
        if error:
            errors.append(error)
        clinical[field] = readings
        counts[field] = len(readings or [])
    return clinical, errors, counts


def _assess():
    """Run the ML layer on the form values. Never raises; returns a dict for display.

    All problems are reported together: text that is not a number is listed, and the adapter
    still validates every other field. The model only runs when there are no problems at all.
    """
    clinical, errors, counts = _collect()
    provided = [f for f in feature_adapter.CLINICAL_FIELDS if clinical.get(f) not in (None, [])]
    if not provided and not errors:
        return {"errors": ["The clinical record is empty. Enter the required fields and at least "
                           "one vital sign or laboratory value."]}
    try:
        if errors:
            feature_adapter.build_model_features(clinical)  # validation only; never scored
        else:
            result = feature_adapter.predict_from_clinical(clinical)
    except inference.MortalityInputError as e:
        adapter_errors = e.errors
        if errors:  # unreadable fields were left out, so "no measurements" may not be true
            adapter_errors = [m for m in adapter_errors if not m.startswith("No vital signs")]
        errors += _messages(adapter_errors)
    except inference.ModelLoadError:
        return {"errors": ["The ICU risk model could not be loaded. Check that "
                           "ml/models/aegiscare_mortality_model.pkl is present."], "system": True}
    except Exception:  # noqa: BLE001 - never show a stack trace in the UI
        return {"errors": ["The assessment could not be completed because of an internal error."],
                "system": True}
    if errors:
        return {"errors": list(dict.fromkeys(errors))}
    return {"result": result, "counts": counts, "age_capped": (clinical["age"] or 0) > feature_adapter.AGE_CAP}


# ---------------------------------------------------------------------------
# Validation summary and result
# ---------------------------------------------------------------------------

def _render_errors(outcome):
    errors = outcome["errors"]
    if outcome.get("system"):
        title = "Assessment unavailable"
    else:
        title = f"{len(errors)} problem{'s' if len(errors) != 1 else ''} found – nothing was assessed"
    st.error(f"**{title}**\n" + "\n".join(f"- {m}" for m in errors), icon=":material/error:")


def _format_score(score):
    return "< 0.01" if score < 0.005 else f"{score:.2f}"


def _render_result(outcome, stale=False):
    if stale:
        st.info("The inputs have changed since this was generated. Select **Assess ICU Risk** to update it.",
                icon=":material/update:")
    if outcome is None or "errors" in outcome:
        with card_container("icu_result_empty", muted=True):
            if outcome is None:
                st.markdown("**No assessment yet.** Enter the available ICU measurements and select "
                            "**Assess ICU Risk**.")
                st.caption("The result appears here. Nothing is stored or sent anywhere.")
            else:
                st.markdown("**Not assessed.** Fix the problems listed above the **Assess ICU Risk** button.")
        return

    result, counts = outcome["result"], outcome["counts"]
    elevated = result["risk_category"] == inference.CATEGORY_ELEVATED
    threshold = result["decision_threshold"]
    completeness = _completeness(result, counts)
    with card_container("icu_result", tone="warning" if elevated else "info"):
        score_col, category_col = st.columns(2, gap="large")
        with score_col:
            label("ICU risk score")
            st.metric("ICU risk score", _format_score(result["risk_score"]), label_visibility="collapsed")
            st.caption("Model-derived ICU risk score")
        with category_col:
            summary_block(
                "Risk category", result["risk_category"].upper(),
                detail=f"{'At or above' if elevated else 'Below'} model threshold · {threshold:.3f}",
                tone="warning" if elevated else None,
            )
            st.caption(f"Demonstration model category: score {'at or above' if elevated else 'below'} the "
                       "model decision threshold. Not a clinical risk classification.")

        st.divider()
        key_value_grid([
            ("Model", result["model_name"]),
            ("Model version", f"v{result['model_version']}"),
            ("Score type", "Uncalibrated model risk score"),
            ("Decision threshold", f"{threshold:.3f}"),
            ("Data completeness", f"{len(completeness['recorded'])} / {len(MEASUREMENTS)} recorded"),
        ])
        st.caption("Compare the score with the decision threshold; it is not a percentage.")
        _render_completeness_notes(result, completeness, outcome.get("age_capped"))


def _completeness(result, counts):
    """Recorded / missing / single-reading measurements, from the result's features_not_recorded."""
    groups = inference._measurement_groups()
    not_recorded = set(result["features_not_recorded"])
    recorded = [f for f in MEASUREMENTS if groups[f"{f}_missing"][0] not in not_recorded]
    return {
        "recorded": recorded,
        "missing": [f for f in MEASUREMENTS if f not in recorded],
        "single": [f for f in recorded if counts.get(f) == 1],
    }


def _render_completeness_notes(result, completeness, age_capped):
    missing, single = completeness["missing"], completeness["single"]
    routine_missing = [f for f in ROUTINE_MEASUREMENTS if f in missing]
    st.caption(f"Recorded {len(completeness['recorded'])} · Not recorded {len(missing)} · "
               f"Single-reading inputs {len(single)} · {result['n_features_not_recorded']} of "
               f"{len(inference.feature_names())} model inputs filled in by the preprocessing pipeline")
    if routine_missing:
        st.warning(
            "**Limited observation data may affect model output.** Not recorded here but available "
            "for at least 90% of the stays the model learned from: "
            + ", ".join(LABELS[f] for f in routine_missing) + ".",
            icon=":material/data_alert:",
        )
    if single:
        st.info(
            "**Limited observation:** a single reading was entered for "
            + ", ".join(LABELS[f] for f in single) + ". The model learned from 24-hour summaries "
            "(mean, minimum, maximum, variability); one reading gives it less to work with, and its "
            "scores tend to be lower.",
            icon=":material/timeline:",
        )
    if age_capped:
        st.caption("Age above 89 was encoded as 90, as in the training data.")
    if missing:
        st.caption("Not recorded (left blank here and handled by the trained preprocessing pipeline): "
                   + ", ".join(LABELS[f] for f in missing) + ".")
    elif not single:
        st.caption("All measurements were recorded with more than one reading.")


# ---------------------------------------------------------------------------
# Model information and limitations
# ---------------------------------------------------------------------------

def _render_model_info():
    try:
        _, meta = inference.load_model()
    except inference.ModelLoadError:
        st.warning("Model information unavailable: the model files could not be loaded.")
        return
    test = meta["test_metrics"]
    split = meta["split"]
    with st.expander("About this model", icon=":material/info:"):
        with card_container("icu_model", muted=True):
            key_value_grid([
                ("Model", f"{meta['model_name']} · v{meta['model_version']}"),
                ("Training data", "eICU Collaborative Research Database (demo)"),
                ("Target", "ICU mortality (recorded outcome)"),
                ("Observation window", "Vitals first 24 h · labs −6 h to +24 h"),
                ("Held-out test set", f"{split['test_size']} stays · {split['test_positives']} deaths"),
                ("Test ROC-AUC / PR-AUC", f"{test['roc_auc']:.3f} / {test['pr_auc']:.3f}"),
                ("Decision threshold", f"{meta['decision_threshold']:.3f}"),
                ("Score type", "Uncalibrated model risk score"),
                ("Calibration", "None – scores are not probabilities"),
                ("Clinical validation", "None"),
            ])
        st.markdown(
            """
- **Not a probability.** The model was trained with class weighting and never calibrated, so a
  score of 0.37 must not be read as a percentage or as the chance of any outcome.
- **Demonstration categories.** *Lower* and *elevated* only say whether the score is below or at/above
  the model's own decision threshold. No clinical thresholds are defined.
- **Separate from the Deterioration module.** That page shows a simulated score on synthetic
  patients; it does not use this model.
- **Small dataset.** Trained on a demo dataset with 91 deaths; the test set had only 18, so the
  performance figures are uncertain. There is no external validation.
- **Same inputs, same window.** Inputs must use the units shown and come from the stated time window.
  Single readings and missing measurements make the input less like the training data.
- **No treatment guidance.** The score does not recommend, start, stop or change any treatment.
- **Proof of concept.** Built for academic demonstration; it is not a clinical decision-support tool.
"""
        )
