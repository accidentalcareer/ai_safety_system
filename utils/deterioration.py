"""
Deterioration Prediction – the scoring logic (no Streamlit code here).

WHY THIS IS A "SIMULATED DEMONSTRATION" AND NOT A TRAINED MODEL
---------------------------------------------------------------
A supervised machine-learning model (e.g. scikit-learn logistic regression)
needs LABELLED examples: patient observations plus a recorded outcome such
as "deteriorated within the next 24 hours: yes / no". The synthetic records
in this project contain observations but NO outcomes. Inventing labels from
a rule and then training on them would only teach the model to copy that
rule, so no model is trained. `labeled_data_check()` makes this decision
explicit and testable.

WHAT THE DEMO SCORE DOES
------------------------
For each vital sign and a small demo selection of lab tests:

1. deviation = how far the value lies OUTSIDE its range, measured in
   multiples of the range width. Inside the range -> 0.
   Example: range 12–20 (width 8), value 24 -> (24 - 20) / 8 = 0.5
2. worsening = how much the deviation increased over the last 24 hours
   (vital signs only; labs have a single result). Improving -> 0.
3. Both parts are capped at MAX_PART_POINTS (1.0) so a single item with a
   narrow range (e.g. C-reactive protein) cannot dominate the total.
4. points = deviation + worsening. The demo score is the sum of all points.

The ranges are the ILLUSTRATIVE adult ranges already used by the dashboard
(VITAL_SIGNS) and each lab's own reference range from the record. They are
not validated alert thresholds, and the equal weighting of every item is an
arbitrary teaching choice. The score is NOT a probability and NOT a
clinical prediction; it is only a way to show how inputs could be combined
and explained.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.synthetic_patients import PATIENTS, VITAL_SIGNS

RESULT_TYPE_SIMULATED = "Simulated demonstration score"

# Vital signs used by the score (all vitals recorded in the synthetic data).
SCORE_VITALS = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "resp_rate", "temperature"]

# A small demo selection of lab tests. Chosen only so the demo has a few
# common blood tests to show; it is not a validated feature set.
DEMO_LABS = [
    "Lactate",
    "White blood cells",
    "C-reactive protein",
    "Creatinine",
    "Hemoglobin",
    "Platelets",
    "Sodium",
    "Potassium",
]

# The score reads only SCORE_VITALS and DEMO_LABS. It has no medication
# features, so a change to the medication list cannot change the score.
USES_MEDICATION_DATA = False

# Data-entry sanity limits. Values outside these are treated as INVALID
# (e.g. typing errors) and ignored. They are not clinical thresholds.
PLAUSIBLE_LIMITS = {
    "heart_rate": (20, 250),
    "systolic_bp": (40, 260),
    "diastolic_bp": (20, 160),
    "spo2": (50, 100),
    "resp_rate": (2, 60),
    "temperature": (30.0, 43.0),
}

# How far back to look when measuring worsening.
TREND_WINDOW_HOURS = 24

# Cap for each part of an item's points (an arbitrary demo choice, see top).
MAX_PART_POINTS = 1.0

# Name of the outcome column a labelled dataset would need.
OUTCOME_LABEL_KEY = "deterioration_outcome"


# ---------------------------------------------------------------------------
# Labelled-data check
# ---------------------------------------------------------------------------

def labeled_data_check(patients=None):
    """
    Decide whether a supervised model could be trained.

    Returns (is_available, reason). Training is only considered possible if
    patient records contain an outcome label, with both outcomes present.
    """
    patients = PATIENTS if patients is None else patients
    labels = [p.get(OUTCOME_LABEL_KEY) for p in patients if p.get(OUTCOME_LABEL_KEY) is not None]
    if not labels:
        return False, (
            f"None of the {len(patients)} synthetic patient records contains an outcome label "
            f"('{OUTCOME_LABEL_KEY}'), so there is nothing for a model to learn from."
        )
    if len(set(labels)) < 2:
        return False, "Only one outcome class is present, so a classifier cannot be trained."
    return True, f"{len(labels)} labelled records found."


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class Contribution:
    """How one vital sign or lab test contributed to the demo score."""

    source: str          # "Vital sign" or "Lab"
    key: str             # e.g. "spo2" or "Lactate"
    label: str           # display name
    value: float         # latest value used
    unit: str
    low: float
    high: float
    deviation: float     # range-widths outside the range (0 if inside)
    worsening: float     # increase in deviation over the trend window (vitals only)
    note: str = ""

    @property
    def points(self):
        return round(self.deviation + self.worsening, 2)


@dataclass
class DemoScoreResult:
    """Everything the demo scorer produced for one patient."""

    score: float = 0.0
    contributions: list = field(default_factory=list)   # list of Contribution
    missing: list = field(default_factory=list)         # human-readable missing-data notes
    invalid: list = field(default_factory=list)         # human-readable invalid-data notes
    items_expected: int = 0
    items_used: int = 0
    latest_time: object = None
    result_type: str = RESULT_TYPE_SIMULATED

    @property
    def coverage(self):
        """Fraction of expected inputs that were available (0–1)."""
        return self.items_used / self.items_expected if self.items_expected else 0.0


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def range_deviation(value, low, high):
    """
    Distance outside [low, high] in multiples of the range width.

    Returns 0 inside the range. Raises ValueError for an empty range.
    """
    if high <= low:
        raise ValueError(f"Invalid range: low ({low}) must be below high ({high})")
    if value < low:
        return (low - value) / (high - low)
    if value > high:
        return (value - high) / (high - low)
    return 0.0


def clean_vitals(vitals):
    """
    Validate and tidy a vital-signs table before scoring.

    - `vitals` must be a pandas DataFrame with a "timestamp" column.
    - Missing vital columns are added as empty (all missing).
    - Non-numeric entries and values outside PLAUSIBLE_LIMITS become NaN
      (missing) and are reported as invalid.
    - Rows without a valid timestamp are dropped; rows are sorted by time.

    Returns (cleaned DataFrame, list of invalid-data notes).
    """
    if not isinstance(vitals, pd.DataFrame):
        raise TypeError("vitals must be a pandas DataFrame")
    if "timestamp" not in vitals.columns:
        raise ValueError("vitals must have a 'timestamp' column")

    df = vitals.copy()
    invalid = []

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_times = int(df["timestamp"].isna().sum())
    if bad_times:
        invalid.append(f"{bad_times} reading(s) without a valid time were ignored.")
        df = df.dropna(subset=["timestamp"])

    for key in SCORE_VITALS:
        if key not in df.columns:
            df[key] = np.nan
            continue
        original = df[key]
        numeric = pd.to_numeric(original, errors="coerce")
        not_numeric = numeric.isna() & original.notna()
        low, high = PLAUSIBLE_LIMITS[key]
        implausible = numeric.notna() & ~numeric.between(low, high)
        label = VITAL_SIGNS[key]["label"]
        if not_numeric.any():
            invalid.append(f"{label}: {int(not_numeric.sum())} non-numeric value(s) ignored.")
        if implausible.any():
            examples = ", ".join(f"{v:g}" for v in numeric[implausible].head(3))
            invalid.append(
                f"{label}: {int(implausible.sum())} value(s) outside the data-entry limits "
                f"{low}–{high} ignored ({examples})."
            )
        df[key] = numeric.mask(implausible).astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, invalid


def vital_features(df, key, window_hours=TREND_WINDOW_HOURS):
    """
    Latest value and 24-hour comparison value for one vital sign.

    `df` must already be cleaned with clean_vitals(). Returns a dict with
    "latest", "latest_time", "earlier" (None if no earlier reading in the
    window) and "n_readings", or None if the vital has no readings at all.
    """
    series = df.set_index("timestamp")[key].dropna()
    if series.empty:
        return None
    latest_time = series.index[-1]
    window = series[series.index >= latest_time - pd.Timedelta(hours=window_hours)]
    earlier = window.iloc[0] if len(window) > 1 else None
    return {
        "latest": float(series.iloc[-1]),
        "latest_time": latest_time,
        "earlier": None if earlier is None else float(earlier),
        "n_readings": int(len(series)),
    }


def lab_inputs(patient):
    """
    Pull the DEMO_LABS results (value, unit, range) out of a patient record.

    Returns (dict test -> (value, unit, low, high), list of invalid notes).
    Labs with a non-numeric value or a broken range are reported as invalid.
    """
    found, invalid = {}, []
    for entry in (patient or {}).get("labs") or []:
        try:
            test, value, unit, low, high = entry
        except (TypeError, ValueError):
            invalid.append(f"A lab entry could not be read: {entry!r}")
            continue
        if test not in DEMO_LABS:
            continue
        try:
            value, low, high = float(value), float(low), float(high)
        except (TypeError, ValueError):
            invalid.append(f"{test}: value or range is not a number, ignored.")
            continue
        if np.isnan(value) or high <= low:
            invalid.append(f"{test}: missing value or invalid reference range, ignored.")
            continue
        found[test] = (value, unit, low, high)
    return found, invalid


# ---------------------------------------------------------------------------
# The demo score
# ---------------------------------------------------------------------------

def _capped(points):
    return min(points, MAX_PART_POINTS)


def compute_demo_score(vitals, patient, include_labs=True):
    """
    Calculate the simulated demonstration score for one patient.

    vitals:  DataFrame of readings (see clean_vitals for accepted input)
    patient: patient record dict (only "labs" is used); may be None
    """
    df, invalid = clean_vitals(vitals)
    result = DemoScoreResult(invalid=invalid)

    if df.empty:
        result.missing.append("No vital-sign readings are available.")

    for key in SCORE_VITALS:
        info = VITAL_SIGNS[key]
        result.items_expected += 1
        feats = vital_features(df, key) if not df.empty else None
        if feats is None:
            result.missing.append(f"{info['label']}: no valid readings – not included in the score.")
            continue

        if feats["latest_time"] < df["timestamp"].iloc[-1]:
            result.missing.append(
                f"{info['label']}: missing at the latest observation time – using the earlier "
                f"reading from {feats['latest_time']:%d %b %H:%M}."
            )

        deviation = _capped(range_deviation(feats["latest"], info["low"], info["high"]))
        worsening, note = 0.0, ""
        if feats["earlier"] is None:
            note = "Only one reading in the last 24 h – trend not assessed."
            result.missing.append(f"{info['label']}: {note}")
        else:
            earlier_dev = _capped(range_deviation(feats["earlier"], info["low"], info["high"]))
            worsening = max(0.0, deviation - earlier_dev)

        if result.latest_time is None or feats["latest_time"] > result.latest_time:
            result.latest_time = feats["latest_time"]
        result.items_used += 1
        result.contributions.append(Contribution(
            source="Vital sign", key=key, label=info["label"], value=feats["latest"],
            unit=info["unit"], low=info["low"], high=info["high"],
            deviation=round(deviation, 2), worsening=round(worsening, 2), note=note,
        ))

    if include_labs:
        labs, lab_invalid = lab_inputs(patient)
        result.invalid.extend(lab_invalid)
        for test in DEMO_LABS:
            result.items_expected += 1
            if test not in labs:
                result.missing.append(f"{test}: not recorded – not included in the score.")
                continue
            value, unit, low, high = labs[test]
            result.items_used += 1
            result.contributions.append(Contribution(
                source="Lab", key=test, label=test, value=value, unit=unit, low=low, high=high,
                deviation=round(_capped(range_deviation(value, low, high)), 2), worsening=0.0,
                note="Single result – no trend available.",
            ))

    result.contributions.sort(key=lambda c: c.points, reverse=True)
    result.score = round(sum(c.points for c in result.contributions), 2)
    return result


def score_history(vitals, step=1):
    """
    Vital-sign part of the demo score at each reading time.

    At each time point only the readings up to that time are used, so the
    chart shows how the score would have looked as data arrived.
    Returns a DataFrame with "timestamp" and "score".
    """
    df, _ = clean_vitals(vitals)
    rows = []
    for end in range(0, len(df), step):
        upto = df.iloc[: end + 1]
        score = compute_demo_score(upto, None, include_labs=False).score
        rows.append({"timestamp": upto["timestamp"].iloc[-1], "score": score})
    return pd.DataFrame(rows, columns=["timestamp", "score"])


def contributions_table(result):
    """Contributions as a DataFrame for display."""
    rows = [
        {
            "Factor": c.label,
            "Type": c.source,
            "Latest value": f"{c.value:g} {c.unit}",
            "Range used": f"{c.low:g}–{c.high:g}",
            "Outside range": c.deviation,
            "Worsening (24 h)": c.worsening,
            "Points": c.points,
            "Note": c.note,
        }
        for c in result.contributions
    ]
    return pd.DataFrame(rows)
