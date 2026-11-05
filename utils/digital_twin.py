"""
Patient Digital Twin – the integration logic (no Streamlit code here).

A "twin" is a private, editable copy of ONE synthetic patient record. The
user can add or remove medications and change the latest vital signs or
lab values. After each change the twin re-runs the EXISTING modules:

- Drug Interaction Checker (utils/drug_interactions.py): documented rules.
  Rules match on drug NAMES only. Patient values listed in a rule's
  "context_labs" / "context_vitals" are shown next to a match but never
  decide whether it matches.
- Deterioration demo score (utils/deterioration.py): simulated logic.
  It uses vital signs and DEMO_LABS only (USES_MEDICATION_DATA is False).

So the twin reassesses a module only when that module actually uses the
changed information, and every timeline event records which modules were
reassessed, why, and where each output came from.

There is no evaluated (trained and validated) model in this project – see
labeled_data_check() – so no output is ever labelled as one.

Everything here is a synthetic college demonstration. Nothing in this
module recommends or performs a real medication change.
"""

import copy
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from data.synthetic_patients import (
    VITAL_SIGNS,
    generate_vitals,
    get_patient,
    lab_results_table,
    medications_table,
)
from utils.deterioration import (
    DEMO_LABS,
    PLAUSIBLE_LIMITS,
    RESULT_TYPE_SIMULATED,
    SCORE_VITALS,
    USES_MEDICATION_DATA,
    compute_demo_score,
)
from utils.drug_interactions import (
    RESULT_TYPE_RULE,
    check_interactions,
    display_name,
    is_known_drug,
    normalize_name,
    patient_medication_names,
)

MODULE_INTERACTIONS = "Drug Interaction Checker"
MODULE_DETERIORATION = "Deterioration demo score"

# Where an output came from (shown in the timeline).
SOURCE_RULE = RESULT_TYPE_RULE                 # "Documented rule"
SOURCE_SIMULATED = RESULT_TYPE_SIMULATED       # "Simulated demonstration score"
SOURCE_STATE_EDIT = "Simulated logic (demo state change)"

STATUS_CURRENT = "Current"
STATUS_NEEDS_REASSESSMENT = "Needs reassessment"

# Text used when a module was not reassessed.
NOT_REASSESSED = "not reassessed"

_ADDED_DEFAULT = "Not specified (simulated)"


class TwinInputError(ValueError):
    """Invalid or no-op user input. The twin's state is left unchanged."""


@dataclass
class TwinEvent:
    """One entry in a patient's timeline."""

    event_id: int
    timestamp: datetime
    patient_id: str
    change_type: str
    item: str
    previous_value: str
    new_value: str
    modules_reassessed: list
    reasons: list            # why each module was / was not reassessed
    status: str              # resulting assessment status
    sources: list            # where the outputs came from


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def format_vital(key, value):
    """'82 bpm', or 'missing' for an empty reading."""
    if _is_missing(value):
        return "missing"
    return f"{value:g} {VITAL_SIGNS[key]['unit']}"


def _pair(interaction):
    return f"{display_name(interaction.drug_a)} + {display_name(interaction.drug_b)}"


def _pairs(result):
    return {_pair(i) for i in result.interactions}


def _to_number(value, what):
    """Convert user input to a float or raise TwinInputError."""
    if isinstance(value, bool):
        raise TwinInputError(f"{what}: '{value}' is not a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TwinInputError(f"{what}: '{value}' is not a number.") from None
    if math.isinf(number):
        raise TwinInputError(f"{what}: the value must be a finite number.")
    return number


# ---------------------------------------------------------------------------
# One patient's twin
# ---------------------------------------------------------------------------

class PatientTwin:
    """Simulated state of one synthetic patient plus its event history."""

    def __init__(self, patient_id, clock=datetime.now):
        get_patient(patient_id)  # raises KeyError for an unknown id
        self.patient_id = patient_id
        self._clock = clock
        self.events = []
        self._load_original()
        self._record(
            "Twin created", "Synthetic record", "—", "Original sample values",
            modules=[MODULE_INTERACTIONS, MODULE_DETERIORATION],
            reasons=["Initial assessment of the original synthetic record."],
            sources=[SOURCE_RULE, SOURCE_SIMULATED],
        )

    # --- state -------------------------------------------------------------

    def _load_original(self):
        """(Re)load the original sample values and assess them."""
        # Deep copy: edits must never leak into the shared PATIENTS data.
        self.record = copy.deepcopy(get_patient(self.patient_id))
        self.vitals = generate_vitals(self.patient_id)
        self.interactions = check_interactions(patient_medication_names(self.record))
        self.deterioration = compute_demo_score(self.vitals, self.record)
        self.deterioration_status = STATUS_CURRENT
        self.deterioration_note = ""

    @property
    def medications(self):
        return self.record.get("medications") or []

    @property
    def medication_names(self):
        return [name for name in patient_medication_names(self.record) if name]

    @property
    def labs(self):
        return self.record.get("labs") or []

    def latest_vitals(self):
        """Latest vital-sign reading (a Series; missing values are NaN)."""
        return self.vitals.iloc[-1]

    def medications_table(self):
        return medications_table(self.medications)

    def lab_table(self):
        return lab_results_table(self.labs)

    def lab_value(self, test):
        for name, value, *_ in self.labs:
            if name == test:
                return value
        return None

    def status_summary(self):
        n = len(self.interactions.interactions)
        rules = f"{n} documented rule match{'es' if n != 1 else ''}"
        return (
            f"Interactions: {rules} · Demo score {self.deterioration.score:g} "
            f"({self.deterioration_status})"
        )

    # --- shared reassessment steps -------------------------------------------

    def _rerun_interactions(self):
        """Re-run the checker and describe what changed."""
        before = _pairs(self.interactions)
        self.interactions = check_interactions(patient_medication_names(self.record))
        after = _pairs(self.interactions)
        parts = []
        if after - before:
            parts.append("new match: " + ", ".join(sorted(after - before)))
        if before - after:
            parts.append("no longer matched: " + ", ".join(sorted(before - after)))
        return "; ".join(parts) or "rule matches unchanged"

    def _rescore(self):
        """Recompute the demo score; returns 'old → new'."""
        before = self.deterioration.score
        self.deterioration = compute_demo_score(self.vitals, self.record)
        self.deterioration_status = STATUS_CURRENT
        self.deterioration_note = ""
        return f"{before:g} → {self.deterioration.score:g}"

    def _interactions_using(self, attribute, key):
        """Matched interactions whose rule lists `key` as patient context."""
        return [i for i in self.interactions.interactions if key in getattr(i, attribute)]

    def _record(self, change_type, item, previous, new, modules, reasons, sources):
        event = TwinEvent(
            event_id=len(self.events) + 1,
            timestamp=self._clock(),
            patient_id=self.patient_id,
            change_type=change_type,
            item=item,
            previous_value=previous,
            new_value=new,
            modules_reassessed=list(modules),
            reasons=list(reasons),
            status=self.status_summary(),
            sources=list(sources),
        )
        self.events.append(event)
        return event

    # --- medication workflow ---------------------------------------------------

    def add_medication(self, name, dose=None, route=None, frequency=None):
        """Add a medication to the simulated list and reassess."""
        if not isinstance(name, str) or not name.strip():
            raise TwinInputError("Enter a medication name.")
        typed = " ".join(name.split())
        canonical = normalize_name(typed)
        for existing in self.medication_names:
            if normalize_name(existing) == canonical:
                same = "" if existing.lower() == typed.lower() else f" (as {existing})"
                raise TwinInputError(f"{typed} is already on the simulated medication list{same}.")

        previous = ", ".join(self.medication_names) or "none"
        self.record.setdefault("medications", []).append({
            "name": display_name(typed),
            "dose": dose or _ADDED_DEFAULT,
            "route": route or _ADDED_DEFAULT,
            "frequency": frequency or _ADDED_DEFAULT,
            "indication": "Simulated demo addition",
        })
        return self._after_medication_change("Medication added", display_name(typed), previous, typed)

    def remove_medication(self, name):
        """Remove a medication from the simulated list and reassess."""
        canonical = normalize_name(name)
        meds = self.medications
        index = next(
            (i for i, med in enumerate(meds) if normalize_name(med.get("name")) == canonical),
            None,
        )
        if canonical is None or index is None:
            raise TwinInputError(f"'{name}' is not on the simulated medication list.")
        previous = ", ".join(self.medication_names)
        removed = meds.pop(index)
        return self._after_medication_change("Medication removed", removed["name"], previous, None)

    def _after_medication_change(self, change_type, item, previous, added_name):
        reasons = [f"{MODULE_INTERACTIONS}: medication list changed, so the documented "
                   f"rules were re-checked ({self._rerun_interactions()})."]
        if added_name and not is_known_drug(added_name):
            reasons.append(
                f"{MODULE_INTERACTIONS}: '{added_name}' is not in the sample dataset, so it "
                "could not be checked. This does not mean it is safe."
            )
        modules, sources = [MODULE_INTERACTIONS], [SOURCE_RULE, SOURCE_STATE_EDIT]

        # Notify the deterioration module that medication exposure changed.
        if USES_MEDICATION_DATA:
            change = self._rescore()
            modules.append(MODULE_DETERIORATION)
            sources.append(SOURCE_SIMULATED)
            reasons.append(f"{MODULE_DETERIORATION}: recomputed with the new medication list ({change}).")
        else:
            self.deterioration_status = STATUS_NEEDS_REASSESSMENT
            self.deterioration_note = (
                f"Medication exposure changed ({change_type.lower()}: {item}). The current "
                "simulated score does not use medication data – only vital signs and a few lab "
                "values – so it was not recomputed and cannot reflect this change."
            )
            reasons.append(f"{MODULE_DETERIORATION}: {NOT_REASSESSED} – marked as needing "
                           "reassessment because the current score does not use medication data.")

        new = ", ".join(self.medication_names) or "none"
        return self._record(change_type, item, previous, new, modules, reasons, sources)

    # --- vital-sign and lab workflow ---------------------------------------------

    def update_vital(self, key, value):
        """
        Change the LATEST reading of one vital sign (None = mark as missing).

        Values outside the deterioration module's data-entry limits are
        rejected, so the twin never holds an impossible value.
        """
        if key not in SCORE_VITALS:
            supported = ", ".join(VITAL_SIGNS[k]["label"] for k in SCORE_VITALS)
            raise TwinInputError(f"'{key}' is not a supported vital sign. Supported: {supported}.")
        label = VITAL_SIGNS[key]["label"]
        new_value = np.nan if _is_missing(value) else _to_number(value, label)
        if not math.isnan(new_value):
            low, high = PLAUSIBLE_LIMITS[key]
            if not low <= new_value <= high:
                raise TwinInputError(
                    f"{label}: {new_value:g} is outside the data-entry limits {low}–{high} used "
                    "by the deterioration module, so it was not applied."
                )

        row = self.vitals.index[-1]
        old_value = self.vitals.at[row, key]
        if (_is_missing(old_value) and math.isnan(new_value)) or old_value == new_value:
            raise TwinInputError(f"{label} is already {format_vital(key, old_value)} – nothing changed.")
        self.vitals.at[row, key] = new_value

        change = self._rescore()
        modules = [MODULE_DETERIORATION]
        sources = [SOURCE_STATE_EDIT, SOURCE_SIMULATED]
        reasons = [f"{MODULE_DETERIORATION}: {label} is one of its inputs, so it was "
                   f"recomputed ({change})."]
        reasons.append(self._context_reassessment("context_vitals", key, label, modules, sources))
        return self._record(
            "Vital sign changed", label, format_vital(key, old_value),
            format_vital(key, new_value), modules, reasons, sources,
        )

    def update_lab(self, test, value):
        """Change the value of a lab test that exists in this record."""
        entries = self.labs
        index = next((i for i, entry in enumerate(entries) if entry[0] == test), None)
        if index is None:
            raise TwinInputError(f"'{test}' is not recorded for this patient, so it cannot be changed.")
        new_value = _to_number(value, test)
        if math.isnan(new_value):
            raise TwinInputError(f"{test}: enter a value (a lab result cannot be blank).")
        if new_value < 0:
            raise TwinInputError(f"{test}: a lab value cannot be negative.")

        name, old_value, unit, low, high = entries[index]
        if old_value == new_value:
            raise TwinInputError(f"{test} is already {old_value:g} {unit} – nothing changed.")
        entries[index] = (name, new_value, unit, low, high)

        modules, sources = [], [SOURCE_STATE_EDIT]
        if test in DEMO_LABS:
            change = self._rescore()
            modules.append(MODULE_DETERIORATION)
            sources.append(SOURCE_SIMULATED)
            reasons = [f"{MODULE_DETERIORATION}: {test} is one of its demo lab inputs, so it was "
                       f"recomputed ({change})."]
        else:
            reasons = [f"{MODULE_DETERIORATION}: {NOT_REASSESSED} – {test} is not one of the lab "
                       "values the demo score uses."]
        reasons.append(self._context_reassessment("context_labs", test, test, modules, sources))
        return self._record(
            "Lab value changed", test, f"{old_value:g} {unit}", f"{new_value:g} {unit}",
            modules, reasons, sources,
        )

    def _context_reassessment(self, attribute, key, label, modules, sources):
        """
        Re-run the interaction checker only if a matched rule lists the changed
        item as patient context. Returns the reason text for the timeline.
        """
        affected = self._interactions_using(attribute, key)
        if not affected:
            return (f"{MODULE_INTERACTIONS}: {NOT_REASSESSED} – no matched rule lists "
                    f"{label} as relevant context, and rules match on drug names only.")
        self._rerun_interactions()
        modules.append(MODULE_INTERACTIONS)
        sources.append(SOURCE_RULE)
        pairs = ", ".join(_pair(i) for i in affected)
        return (f"{MODULE_INTERACTIONS}: re-evaluated because the rule for {pairs} lists {label} "
                "as relevant patient context; the displayed context value was refreshed. Rule "
                "matches depend on drug names only, so they did not change.")

    # --- manual reassessment and reset ---------------------------------------------

    def reassess(self):
        """Re-run both modules on the current simulated state."""
        was_pending = self.deterioration_status == STATUS_NEEDS_REASSESSMENT
        interaction_change = self._rerun_interactions()
        score_change = self._rescore()
        reasons = [
            f"{MODULE_INTERACTIONS}: re-run on request ({interaction_change}).",
            f"{MODULE_DETERIORATION}: recomputed on request ({score_change}).",
        ]
        if was_pending and not USES_MEDICATION_DATA:
            reasons.append(
                f"{MODULE_DETERIORATION}: the pending medication change cannot affect this score "
                "because it has no medication inputs."
            )
        return self._record(
            "Manual reassessment", "All modules", "—", "—",
            [MODULE_INTERACTIONS, MODULE_DETERIORATION], reasons,
            [SOURCE_RULE, SOURCE_SIMULATED],
        )

    def reset(self):
        """Restore the original sample values. The timeline is kept for audit."""
        changes = 0
        for event in reversed(self.events):
            if event.change_type in ("Twin created", "Reset"):
                break
            changes += event.change_type != "Manual reassessment"
        self._load_original()
        return self._record(
            "Reset", "Whole simulated state", f"{changes} simulated change(s) since the last reset",
            "Original sample values", [MODULE_INTERACTIONS, MODULE_DETERIORATION],
            ["All modules re-run on the original synthetic record."],
            [SOURCE_RULE, SOURCE_SIMULATED],
        )


# ---------------------------------------------------------------------------
# One twin per patient
# ---------------------------------------------------------------------------

class TwinRegistry:
    """Keeps one independent PatientTwin per patient id."""

    def __init__(self, clock=datetime.now):
        self._clock = clock
        self._twins = {}

    def get(self, patient_id):
        if patient_id not in self._twins:
            self._twins[patient_id] = PatientTwin(patient_id, clock=self._clock)
        return self._twins[patient_id]

    def __contains__(self, patient_id):
        return patient_id in self._twins


TIMELINE_COLUMNS = [
    "#", "Time", "Patient ID", "Change", "Item", "Previous", "New",
    "Modules reassessed", "Resulting status", "Output source", "Why",
]


def timeline_table(events):
    """Events as a DataFrame, newest first."""
    rows = [
        {
            "#": e.event_id,
            "Time": e.timestamp,
            "Patient ID": e.patient_id,
            "Change": e.change_type,
            "Item": e.item,
            "Previous": e.previous_value,
            "New": e.new_value,
            "Modules reassessed": ", ".join(e.modules_reassessed) or "none",
            "Resulting status": e.status,
            "Output source": ", ".join(e.sources),
            "Why": " ".join(e.reasons),
        }
        for e in reversed(events)
    ]
    return pd.DataFrame(rows, columns=TIMELINE_COLUMNS)
