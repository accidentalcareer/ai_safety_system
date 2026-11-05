"""Inference interface for the saved AEGISCARE ICU-mortality model.

Usage:
    from inference import predict_mortality   # or `from ml.inference import ...` from the repo root
    result = predict_mortality(patient_features)   # dict / pandas Series / 1-row DataFrame

Proof of concept for a college project - NOT clinically validated, NOT for patient care.

* All preprocessing that was learned during training (median imputation, constant-feature filter,
  scaling) lives inside the saved sklearn Pipeline and is reused as-is. This module only checks
  that the input has the schema produced by `preprocessing.py` before handing it to the pipeline.
* The output is an uncalibrated MODEL RISK SCORE (the model was trained with class_weight=
  'balanced'), not a probability of death.
* Risk categories are demonstration categories defined only by the model's decision threshold
  (stored in model_metadata.json); they are not clinical risk classes.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (  # noqa: E402
    APERIODIC_VITALS, LAB_NAMES, LAB_STATS, PERIODIC_VITALS, PLAUSIBLE_RANGES, VITAL_STATS,
)

ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "models" / "aegiscare_mortality_model.pkl"
METADATA_PATH = ML_DIR / "models" / "model_metadata.json"
EXPECTED_N_FEATURES = 126

SCORE_TYPE = "uncalibrated model risk score (not a probability)"
DISCLAIMER = ("Proof-of-concept model trained on the eICU demo dataset. Not clinically validated; "
              "not for patient-care decisions. The score ranks patients relative to the model's "
              "decision threshold and is not a probability of death.")

# Demonstration categories: the only cut-point used is the model's own decision threshold
# (F1-maximising threshold on out-of-fold training predictions, see model_metadata.json).
CATEGORY_LOWER = "lower"
CATEGORY_ELEVATED = "elevated"

VITALS = PERIODIC_VITALS + APERIODIC_VITALS
STATIC_NUMERIC = ["age", "admissionweight", "admissionheight"]
ONE_HOT_GROUPS = ["gender_", "unittype_", "unitstaytype_"]  # gender_ includes gender_missing


class ModelLoadError(RuntimeError):
    """The model or its metadata is missing, unreadable, or inconsistent."""


class MortalityInputError(ValueError):
    """The patient record cannot be scored. `errors` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid patient record:\n  - " + "\n  - ".join(errors))


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, dict]:
    """Load (pipeline, metadata) once and check they agree with each other."""
    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(METADATA_PATH.read_text())
    except FileNotFoundError as e:
        raise ModelLoadError(f"Model file not found: {e.filename}") from e
    except Exception as e:
        raise ModelLoadError(f"Could not load model artefacts: {type(e).__name__}: {e}") from e

    features = meta.get("feature_names", [])
    if len(features) != EXPECTED_N_FEATURES:
        raise ModelLoadError(f"Metadata lists {len(features)} features, expected {EXPECTED_N_FEATURES}")
    if list(getattr(model, "feature_names_in_", [])) != features:
        raise ModelLoadError("Model feature names do not match model_metadata.json")
    if not hasattr(model, "predict_proba"):
        raise ModelLoadError("Saved model has no predict_proba method")
    if not 0 < meta.get("decision_threshold", -1) < 1:
        raise ModelLoadError("Metadata has no valid decision_threshold")
    return model, meta


def feature_names() -> list[str]:
    """The 126 input features, in the order the model expects."""
    return list(load_model()[1]["feature_names"])


# --------------------------------------------------------------------------- validation

def _to_record(patient_features: Any) -> dict[str, Any]:
    if isinstance(patient_features, pd.DataFrame):
        if len(patient_features) != 1:
            raise MortalityInputError([f"Expected one patient (1 row), got {len(patient_features)} rows"])
        patient_features = patient_features.iloc[0]
    if isinstance(patient_features, pd.Series):
        patient_features = patient_features.to_dict()
    if not isinstance(patient_features, Mapping):
        raise MortalityInputError([f"Expected a dict, pandas Series or 1-row DataFrame, "
                                   f"got {type(patient_features).__name__}"])
    if not patient_features:
        raise MortalityInputError(["Patient record is empty"])
    return dict(patient_features)


def _is_blank(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, Real) and math.isnan(value))


def _measurement_groups() -> dict[str, tuple[str, list[str]]]:
    """indicator column -> (primary column, all value columns) for every measured variable."""
    groups = {f"{v}_missing": (f"{v}_mean", [f"{v}_{s}" for s in VITAL_STATS]) for v in VITALS}
    groups |= {f"{lab}_missing": (f"{lab}_first", [f"{lab}_{s}" for s in LAB_STATS]) for lab in LAB_NAMES}
    groups |= {f"{s}_missing": (s, [s]) for s in STATIC_NUMERIC}
    return groups


def validate_features(patient_features: Any) -> pd.DataFrame:
    """Check a patient record and return it as a 1-row DataFrame in model column order.

    Rules (all derived from how `preprocessing.py` builds the training data):
    * every one of the 126 features must be present, and no unknown keys are allowed;
    * indicator / one-hot / APACHE-flag columns must be 0 or 1 (never blank);
      exactly one column is set in each one-hot group;
    * measured values may be None/NaN only when not recorded, and `<var>_missing` must agree;
      the saved pipeline then imputes them exactly as in training;
    * recorded values must be finite numbers inside the plausibility ranges used for cleaning,
      with min <= first/mean <= max and std >= 0;
    * at least one vital sign or lab value must be recorded.
    """
    record = _to_record(patient_features)
    features = feature_names()
    errors: list[str] = []

    missing = [f for f in features if f not in record]
    unknown = sorted(str(k) for k in record if k not in set(features))
    if missing:
        errors.append(f"{len(missing)} required feature(s) missing: {missing}")
    if unknown:
        errors.append(f"{len(unknown)} unknown feature(s): {unknown}")
    if errors:
        raise MortalityInputError(errors)

    values: dict[str, float] = {}
    for f in features:
        v = record[f]
        if isinstance(v, (bool, np.bool_)):
            v = int(v)
        if _is_blank(v):
            values[f] = np.nan
        elif not isinstance(v, Real) or not math.isfinite(v):
            errors.append(f"{f}: expected a finite number or None, got {v!r}")
        else:
            values[f] = float(v)
    if errors:
        raise MortalityInputError(errors)

    groups = _measurement_groups()
    value_cols = {c for _, cols in groups.values() for c in cols}
    binary_cols = [f for f in features if f not in value_cols]
    for f in binary_cols:
        if values[f] not in (0.0, 1.0):
            errors.append(f"{f}: must be 0 or 1, got {record[f]!r}")
    for prefix in ONE_HOT_GROUPS:
        cols = [f for f in binary_cols if f.startswith(prefix) and not f.endswith("_missing")
                or f == f"{prefix}missing"]
        if sum(values[c] == 1.0 for c in cols) != 1:
            errors.append(f"exactly one of {cols} must be 1")

    for indicator, (primary, cols) in groups.items():
        flagged = values[indicator] == 1.0
        recorded = [c for c in cols if not np.isnan(values[c])]
        if flagged and recorded:
            errors.append(f"{indicator}=1 but values are given for {recorded}")
        if not flagged and np.isnan(values[primary]):
            errors.append(f"{primary} is blank but {indicator}=0 (set {indicator}=1 if not recorded)")

    for var, (low, high) in PLAUSIBLE_RANGES.items():
        stats = ["first", "mean", "min", "max"] if var not in STATIC_NUMERIC else [None]
        for stat in stats:
            col = var if stat is None else f"{var}_{stat}"
            if col in values and not np.isnan(values[col]) and not low <= values[col] <= high:
                errors.append(f"{col}={values[col]} is outside the plausible range [{low}, {high}]")
        lo, hi = values.get(f"{var}_min", np.nan), values.get(f"{var}_max", np.nan)
        for mid in (f"{var}_first", f"{var}_mean"):
            m = values.get(mid, np.nan)
            if not np.isnan(m) and not (np.isnan(lo) or np.isnan(hi)) and not lo <= m <= hi:
                errors.append(f"{mid}={m} is not between {var}_min={lo} and {var}_max={hi}")
        if values.get(f"{var}_std", 0) < 0:
            errors.append(f"{var}_std must be >= 0")

    measured = [c for v in VITALS + list(LAB_NAMES) for c in groups[f"{v}_missing"][1]]
    if all(np.isnan(values[c]) for c in measured):
        errors.append("No vital signs or laboratory values recorded; a risk score would rest "
                      "entirely on imputed values")
    if errors:
        raise MortalityInputError(errors)
    return pd.DataFrame([values], columns=features)


# --------------------------------------------------------------------------- prediction

def risk_category(score: float, threshold: float) -> str:
    """Demonstration category: 'elevated' if score >= the model decision threshold, else 'lower'."""
    return CATEGORY_ELEVATED if score >= threshold else CATEGORY_LOWER


def predict_mortality(patient_features: Any) -> dict[str, Any]:
    """Score one patient record (the 126 features produced by `preprocessing.py`).

    Raises MortalityInputError for an invalid record and ModelLoadError if the model is unusable.
    """
    model, meta = load_model()
    X = validate_features(patient_features)
    try:
        score = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {type(e).__name__}: {e}") from e
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Model returned an invalid score: {score}")

    threshold = float(meta["decision_threshold"])
    category = risk_category(score, threshold)
    not_recorded = [c for c in X.columns if np.isnan(X.at[0, c])]
    return {
        "risk_score": score,
        "risk_category": category,
        "category_description": (
            f"Demonstration model category: score {'at or above' if category == CATEGORY_ELEVATED else 'below'} "
            f"the model decision threshold ({threshold:.3f}). Not a clinical risk classification."),
        "above_threshold": category == CATEGORY_ELEVATED,
        "decision_threshold": threshold,
        "score_type": SCORE_TYPE,
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "n_features_not_recorded": len(not_recorded),
        "features_not_recorded": not_recorded,
        "disclaimer": DISCLAIMER,
    }
