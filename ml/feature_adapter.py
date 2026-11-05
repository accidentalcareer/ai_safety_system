"""Feature adapter: human-readable clinical input -> the 126 model features used by inference.py.

Usage:
    from feature_adapter import build_model_features, predict_from_clinical
    features = build_model_features({"age": 67, "gender": "Male", "unittype": "MICU",
                                     "unitstaytype": "admit", "intubated": False, "vent": False,
                                     "dialysis": False, "heartrate": [88, 102, 95], "lactate": 3.1})
    result = predict_from_clinical(...same dict...)

Proof of concept for a college project - NOT clinically validated, NOT for patient care.

Every vital / lab field takes ONE reading or a LIST of readings in chronological order, taken in the
same window the model was trained on (vitals: first 24h of the ICU stay; labs: 6h before to 24h
after ICU admission). Readings are summarised with the training code itself
(`preprocessing._aggregate_vitals` -> mean/min/max/std, `LAB_STATS` -> first/min/max), so a single
reading gives min = max = mean (= first) and a blank std, exactly as for a training stay with one
observation. A measurement that is not given stays blank (None) with `<var>_missing` = 1; the
saved pipeline imputes it later, exactly as in training. No clinical value is ever invented.
"""
from __future__ import annotations

import math
import re
import sys
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference  # noqa: E402
from preprocessing import (  # noqa: E402
    APACHE_FLAGS, APERIODIC_VITALS, FAHRENHEIT_RANGE, GENDER_CATEGORIES, KEY, LAB_NAMES, LAB_STATS,
    PERIODIC_VITALS, PLAUSIBLE_RANGES, UNITSTAYTYPE_CATEGORIES, UNITTYPE_CATEGORIES,
    _aggregate_vitals,
)

STATIC_NUMERIC = ["age", "admissionweight", "admissionheight"]
VITALS = PERIODIC_VITALS + APERIODIC_VITALS
LABS = list(LAB_NAMES)
# eICU stores ages above 89 as "> 89", which preprocessing maps to 90; the model never saw > 90.
AGE_CAP = 90
# Defined in preprocessing but constant (never observed) in the training cohort, so the variance
# filter drops it and the model cannot use it.
UNTRAINED_CATEGORIES = {"unitstaytype": {"stepdown/other"}}

CATEGORICAL = {
    "gender": GENDER_CATEGORIES,
    "unittype": UNITTYPE_CATEGORIES,
    "unitstaytype": [c for c in UNITSTAYTYPE_CATEGORIES if c not in UNTRAINED_CATEGORIES["unitstaytype"]],
}
REQUIRED = ["unittype", "unitstaytype", *APACHE_FLAGS]  # no missing indicator exists for these

UNITS = {
    "age": "years", "admissionweight": "kg", "admissionheight": "cm",
    "heartrate": "beats/min", "sao2": "%", "respiration": "breaths/min", "temperature": "°C",
    **{v: "mmHg" for v in VITALS if "systolic" in v or "diastolic" in v or v.endswith("mean")},
    "glucose": "mg/dL", "sodium": "mmol/L", "potassium": "mmol/L", "chloride": "mmol/L",
    "bicarbonate": "mmol/L", "creatinine": "mg/dL", "bun": "mg/dL", "wbc": "x1000/µL",
    "hemoglobin": "g/dL", "platelets": "x1000/µL", "calcium": "mg/dL", "magnesium": "mg/dL",
    "lactate": "mmol/L",
}

CLINICAL_FIELDS = [*STATIC_NUMERIC, *CATEGORICAL, *VITALS, *LABS, *APACHE_FLAGS]


class ClinicalInputError(inference.MortalityInputError):
    """The clinical input cannot be converted. `errors` lists every problem found."""


def _slug(category: str) -> str:
    """One-hot column suffix, same rule as preprocessing.build_patient_features."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", category).strip("_").lower()


def _is_number(v: Any) -> bool:
    return isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)


def _readings(field: str, value: Any, errors: list[str]) -> list[float]:
    """None / [] -> no readings; a number or a list of numbers -> validated readings."""
    if value is None:
        return []
    items = list(value) if isinstance(value, (list, tuple, np.ndarray, pd.Series)) else [value]
    low, high = PLAUSIBLE_RANGES[field]
    out = []
    for v in items:
        if not _is_number(v):
            errors.append(f"{field}: expected a number (or list of numbers), got {v!r}")
        elif not low <= v <= high:
            hint = " - temperature must be in °C" if field == "temperature" and \
                FAHRENHEIT_RANGE[0] <= v <= FAHRENHEIT_RANGE[1] else ""
            errors.append(f"{field}={v} {UNITS[field]} is outside the plausible range [{low}, {high}]{hint}")
        else:
            out.append(float(v))
    return out


def _category(field: str, value: Any, errors: list[str]) -> str | None:
    if value is None:
        return None
    allowed = CATEGORICAL[field]
    if isinstance(value, str):
        match = {c.lower(): c for c in allowed}.get(value.strip().lower())
        if match:
            return match
        if value.strip().lower() in UNTRAINED_CATEGORIES.get(field, set()):
            errors.append(f"{field}={value!r} does not occur in the training data, so the model "
                          f"cannot use it; allowed: {allowed}")
            return None
    errors.append(f"{field}={value!r} is not a supported category; allowed: {allowed}")
    return None


def _flag(field: str, value: Any, errors: list[str]) -> int | None:
    if isinstance(value, (bool, np.bool_)) or value in (0, 1):
        return int(value)
    errors.append(f"{field}: expected True/False (or 1/0), got {value!r}")
    return None


def build_model_features(clinical_input: Mapping[str, Any]) -> dict[str, Any]:
    """Convert clinical input to the 126-feature record expected by inference.predict_mortality.

    Fields (all optional except unittype, unitstaytype, intubated, vent, dialysis):
      age, admissionweight, admissionheight  - number (age > 89 is encoded as 90, as in training;
                                               the string "> 89" is also accepted)
      gender ("Female"/"Male"), unittype, unitstaytype ("admit"/"readmit"/"transfer") - category
      vitals and labs                        - one number or a chronological list of numbers
      intubated, vent, dialysis              - True/False (APACHE first-day flags)
    Blank values are None (missing). Raises ClinicalInputError listing every problem.
    Returns {feature_name: number or None} in the model's feature order.
    """
    if not isinstance(clinical_input, Mapping):
        raise ClinicalInputError([f"Expected a dict of clinical fields, got {type(clinical_input).__name__}"])
    if not clinical_input:
        raise ClinicalInputError(["Clinical input is empty"])
    errors: list[str] = []
    unknown = sorted(str(k) for k in clinical_input if k not in CLINICAL_FIELDS)
    if unknown:
        errors.append(f"unknown field(s): {unknown}; supported fields: {CLINICAL_FIELDS}")
    absent = [f for f in REQUIRED if clinical_input.get(f) is None]
    if absent:
        errors.append(f"required field(s) missing: {absent}")
    get = clinical_input.get
    features: dict[str, Any] = {}

    # ---- static numeric
    for field in STATIC_NUMERIC:
        value = get(field)
        if field == "age" and isinstance(value, str) and value.replace(" ", "") == ">89":
            value = AGE_CAP
        readings = _readings(field, value, errors)
        if len(readings) > 1:
            errors.append(f"{field}: expected a single value, got {len(readings)}")
        number = readings[0] if readings else None
        if field == "age" and number is not None:
            number = min(number, AGE_CAP)
        features[field] = number
        features[f"{field}_missing"] = int(number is None)

    # ---- categoricals: one-hot with the fixed training categories; gender also has gender_missing
    chosen = {field: _category(field, get(field), errors) for field in CATEGORICAL}
    for field, categories in [("gender", GENDER_CATEGORIES), ("unittype", UNITTYPE_CATEGORIES),
                              ("unitstaytype", UNITSTAYTYPE_CATEGORIES)]:
        for cat in categories:
            features[f"{field}_{_slug(cat)}"] = int(chosen[field] == cat)
    features["gender_missing"] = int(chosen["gender"] is None)

    # ---- APACHE first-day flags
    for field in APACHE_FLAGS:
        if get(field) is not None:
            features[f"apache_{field}"] = _flag(field, get(field), errors)

    # ---- vitals: same aggregation function as training (long format, NaN where not measured)
    for variables in (PERIODIC_VITALS, APERIODIC_VITALS):
        rows = [{KEY: 0, **{v: np.nan for v in variables}}]  # keeps the group when nothing measured
        for var in variables:
            rows += [{KEY: 0, var: r} for r in _readings(var, get(var), errors)]
        features.update(_aggregate_vitals(pd.DataFrame(rows), variables).iloc[0].to_dict())

    # ---- labs: first/min/max in chronological order, as in build_lab_features
    for lab in LABS:
        readings = _readings(lab, get(lab), errors)
        stats = (pd.DataFrame({KEY: 0, "v": readings or [np.nan]}).groupby(KEY)["v"]
                 .agg(LAB_STATS).iloc[0])
        for stat in LAB_STATS:
            features[f"{lab}_{stat}"] = stats[stat]
        features[f"{lab}_missing"] = int(not readings)

    if errors:
        raise ClinicalInputError(errors)

    names = inference.feature_names()
    if set(features) != set(names):  # adapter / model schema drift - a bug, not a user error
        raise RuntimeError(f"Adapter schema mismatch: extra {sorted(set(features) - set(names))}, "
                           f"missing {sorted(set(names) - set(features))}")
    record = {}
    for name in names:
        v = features[name]
        record[name] = None if v is None or (isinstance(v, float) and math.isnan(v)) else \
            (int(v) if name.endswith("_missing") or name.startswith(("gender_", "unit", "apache_"))
             else float(v))
    try:
        inference.validate_features(record)  # same checks predict_mortality applies
    except inference.MortalityInputError as e:
        raise ClinicalInputError(e.errors) from e
    return record


def predict_from_clinical(clinical_input: Mapping[str, Any]) -> dict[str, Any]:
    """build_model_features() followed by inference.predict_mortality()."""
    return inference.predict_mortality(build_model_features(clinical_input))
