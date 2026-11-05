"""Tests for ml/feature_adapter.py.

Run:  .venv\\Scripts\\python.exe -m pytest ml/test_feature_adapter.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_adapter as fa  # noqa: E402
import inference as inf  # noqa: E402
import preprocessing as pp  # noqa: E402

BASE = {"unittype": "Med-Surg ICU", "unitstaytype": "admit",
        "intubated": False, "vent": False, "dialysis": False}

COMPLETE = {
    **BASE, "age": 71, "gender": "Female", "admissionweight": 68.5, "admissionheight": 162,
    "heartrate": [96, 104, 110, 99], "sao2": [95, 92, 97], "respiration": [18, 24, 22],
    "temperature": [37.2, 38.1], "systemicsystolic": [118, 104], "systemicdiastolic": [62, 55],
    "systemicmean": [80, 71], "noninvasivesystolic": [122, 108, 115],
    "noninvasivediastolic": [66, 58, 61], "noninvasivemean": [85, 75, 79],
    "glucose": [142, 188], "sodium": [138, 135], "potassium": [4.1, 3.6], "chloride": [101, 99],
    "bicarbonate": [22, 19], "creatinine": [1.2, 1.6], "bun": [24, 31], "wbc": [11.2, 14.8],
    "hemoglobin": [11.9, 10.7], "platelets": [210, 188], "calcium": [8.6, 8.2],
    "magnesium": [2.0, 1.8], "lactate": [2.4, 3.3],
}
PARTIAL = {**BASE, "age": 58, "gender": "Male", "heartrate": 112, "sao2": 91,
           "noninvasivesystolic": [98, 90], "creatinine": 2.3, "lactate": 4.0}


def blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


# --------------------------------------------------------------------------- valid input

def test_complete_input_gives_exactly_the_126_model_features():
    record = fa.build_model_features(COMPLETE)
    assert list(record) == inf.feature_names() and len(record) == 126
    assert not any(record[f] for f in record if f.endswith("_missing"))
    assert record["gender_female"] == 1 and record["gender_male"] == 0 and record["gender_missing"] == 0
    assert record["unittype_med_surg_icu"] == 1 and sum(record[f] for f in record if f.startswith("unittype_")) == 1
    assert record["unitstaytype_admit"] == 1 and record["unitstaytype_stepdown_other"] == 0


def test_aggregation_matches_training_definitions():
    record = fa.build_model_features(COMPLETE)
    hr = COMPLETE["heartrate"]
    assert record["heartrate_mean"] == pytest.approx(np.mean(hr))
    assert record["heartrate_min"] == min(hr) and record["heartrate_max"] == max(hr)
    assert record["heartrate_std"] == pytest.approx(np.std(hr, ddof=1))  # pandas sample std
    # labs: first is the first reading in chronological order, not the minimum
    assert record["glucose_first"] == 142 and record["glucose_min"] == 142 and record["glucose_max"] == 188
    assert record["sodium_first"] == 138 and record["sodium_min"] == 135


def test_single_reading_matches_training_single_observation():
    record = fa.build_model_features(PARTIAL)
    assert record["heartrate_mean"] == record["heartrate_min"] == record["heartrate_max"] == 112
    assert record["heartrate_std"] is None  # one observation -> undefined std, as in training
    assert record["creatinine_first"] == record["creatinine_min"] == record["creatinine_max"] == 2.3


def test_partially_missing_input_and_missing_flags():
    record = fa.build_model_features(PARTIAL)
    given = {"heartrate", "sao2", "noninvasivesystolic", "creatinine", "lactate", "age"}
    groups = inf._measurement_groups()
    for indicator, (_, cols) in groups.items():
        var = indicator.removesuffix("_missing")
        if var in given:
            assert record[indicator] == 0, indicator
            assert not blank(record[cols[0]])
        else:
            assert record[indicator] == 1, indicator
            assert all(blank(record[c]) for c in cols), f"{var}: a value was invented"
    assert record["gender_missing"] == 0


def test_optional_measurements_can_be_omitted_or_none():
    omitted = fa.build_model_features({**BASE, "heartrate": 90})
    explicit = fa.build_model_features({**BASE, "heartrate": 90, "lactate": None, "age": None,
                                        "gender": None, "sodium": [], "temperature": None})
    assert omitted == explicit
    assert omitted["age"] is None and omitted["age_missing"] == 1
    assert omitted["gender_missing"] == 1 and omitted["gender_female"] == omitted["gender_male"] == 0
    assert omitted["lactate_first"] is None and omitted["lactate_missing"] == 1


def test_age_above_89_encoded_as_in_training():
    assert fa.build_model_features({**BASE, "heartrate": 90, "age": 95})["age"] == 90
    assert fa.build_model_features({**BASE, "heartrate": 90, "age": "> 89"})["age"] == 90
    assert fa.build_model_features({**BASE, "heartrate": 90, "age": 89})["age"] == 89


def test_categories_case_insensitive_and_flags_accept_bool_or_int():
    a = fa.build_model_features({**BASE, "heartrate": 90, "unittype": "micu", "gender": "female",
                                 "vent": 1, "dialysis": True})
    assert a["unittype_micu"] == 1 and a["gender_female"] == 1
    assert a["apache_vent"] == 1 and a["apache_dialysis"] == 1 and a["apache_intubated"] == 0


# --------------------------------------------------------------------------- invalid input

@pytest.mark.parametrize("field,value,match", [
    ("heartrate", "fast", "expected a number"),
    ("heartrate", [90, "x"], "expected a number"),
    ("heartrate", np.nan, "expected a number"),
    ("heartrate", True, "expected a number"),
    ("heartrate", 400, "plausible range"),
    ("sao2", 105, "plausible range"),
    ("temperature", 101.3, "must be in °C"),
    ("age", -1, "plausible range"),
    ("age", [60, 61], "single value"),
    ("lactate", float("inf"), "expected a number"),
    ("vent", "yes", "True/False"),
    ("dialysis", 2, "True/False"),
])
def test_invalid_numerical_values_rejected(field, value, match):
    with pytest.raises(fa.ClinicalInputError, match=match):
        fa.build_model_features({**PARTIAL, field: value})


@pytest.mark.parametrize("field,value", [
    ("gender", "Other"), ("gender", "M"), ("unittype", "ICU"), ("unittype", "Emergency"),
    ("unitstaytype", "discharge"), ("unittype", 3),
])
def test_invalid_categorical_values_rejected(field, value):
    with pytest.raises(fa.ClinicalInputError, match=f"{field}="):
        fa.build_model_features({**PARTIAL, field: value})


def test_category_absent_from_training_data_rejected():
    with pytest.raises(fa.ClinicalInputError, match="does not occur in the training data"):
        fa.build_model_features({**PARTIAL, "unitstaytype": "stepdown/other"})


def test_unknown_fields_rejected():
    with pytest.raises(fa.ClinicalInputError, match=r"unknown field\(s\): \['heart_rate', 'spo2'\]"):
        fa.build_model_features({**PARTIAL, "heart_rate": 90, "spo2": 95})


@pytest.mark.parametrize("field", fa.REQUIRED)
def test_required_fields_enforced(field):
    with pytest.raises(fa.ClinicalInputError, match="required field"):
        fa.build_model_features({k: v for k, v in PARTIAL.items() if k != field})


@pytest.mark.parametrize("bad", [{}, None, [], "patient"])
def test_empty_or_wrong_type_rejected(bad):
    with pytest.raises(fa.ClinicalInputError):
        fa.build_model_features(bad)


def test_no_vitals_or_labs_rejected():
    with pytest.raises(fa.ClinicalInputError, match="No vital signs or laboratory values"):
        fa.build_model_features({**BASE, "age": 60, "gender": "Male"})


def test_all_errors_reported_together():
    with pytest.raises(fa.ClinicalInputError) as exc:
        fa.build_model_features({**PARTIAL, "gender": "X", "heartrate": -5, "foo": 1})
    assert len(exc.value.errors) == 3
    assert isinstance(exc.value, inf.MortalityInputError)  # callers can catch one error type


# --------------------------------------------------------------------------- with the model

@pytest.mark.parametrize("clinical", [COMPLETE, PARTIAL], ids=["complete", "partial"])
def test_record_can_be_scored(clinical):
    result = inf.predict_mortality(fa.build_model_features(clinical))
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_category"] in {inf.CATEGORY_LOWER, inf.CATEGORY_ELEVATED}
    assert fa.predict_from_clinical(clinical) == result


def test_repeated_input_is_deterministic():
    records = [fa.build_model_features(dict(COMPLETE)) for _ in range(3)]
    results = [fa.predict_from_clinical(dict(COMPLETE)) for _ in range(3)]
    assert all(r == records[0] for r in records)
    assert all(r == results[0] for r in results)


def test_input_dict_not_mutated():
    clinical = {**COMPLETE, "heartrate": list(COMPLETE["heartrate"])}
    before = repr(clinical)
    fa.build_model_features(clinical)
    assert repr(clinical) == before


# --------------------------------------------------------------------------- round trip

def _raw_clinical_inputs(stays: list[int]) -> dict[int, dict]:
    """Rebuild clinical input from the raw eICU tables with preprocessing's own cleaning steps."""
    log: dict = {}
    pat = pp.read_raw("patient", usecols=[pp.KEY, "age", "gender", "admissionweight",
                                          "admissionheight", "unittype", "unitstaytype"])
    pat = pat[pat[pp.KEY].isin(stays)].set_index(pp.KEY)
    pat["age"] = pd.to_numeric(pat["age"].replace({"> 89": "90"}), errors="coerce")
    pat = pp.mask_implausible(pat, ["age", "admissionweight", "admissionheight"], log)
    aps = pp.read_raw("apacheApsVar", usecols=[pp.KEY, *pp.APACHE_FLAGS])
    aps = aps[aps[pp.KEY].isin(stays)].drop_duplicates(pp.KEY).set_index(pp.KEY)

    lo, hi = pp.VITAL_WINDOW
    vp = pp.read_raw("vitalPeriodic", usecols=[pp.KEY, "observationoffset", *pp.PERIODIC_VITALS])
    vp = vp[vp[pp.KEY].isin(stays) & vp["observationoffset"].between(lo, hi)].drop_duplicates()
    f = vp["temperature"].between(*pp.FAHRENHEIT_RANGE)
    vp.loc[f, "temperature"] = (vp.loc[f, "temperature"] - 32) * 5 / 9
    vp = pp.mask_implausible(vp, pp.PERIODIC_VITALS, log)
    va = pp.read_raw("vitalAperiodic", usecols=[pp.KEY, "observationoffset", *pp.APERIODIC_VITALS])
    va = va[va[pp.KEY].isin(stays) & va["observationoffset"].between(lo, hi)].drop_duplicates()
    va = pp.mask_implausible(va, pp.APERIODIC_VITALS, log)

    lo, hi = pp.LAB_WINDOW
    lab = pp.read_raw("lab", usecols=[pp.KEY, "labresultoffset", "labresultrevisedoffset",
                                      "labname", "labresult", "labresulttext"])
    to_feature = {v: k for k, v in pp.LAB_NAMES.items()}
    lab = lab[lab[pp.KEY].isin(stays) & lab["labname"].isin(to_feature)
              & lab["labresultoffset"].between(lo, hi)].copy()
    lab["feature"] = lab["labname"].map(to_feature)
    c = lab["labresult"].isna()
    lab.loc[c, "labresult"] = pp._parse_censored(lab.loc[c, "labresulttext"])
    lab = (lab.dropna(subset=["labresult"]).sort_values("labresultrevisedoffset")
              .drop_duplicates([pp.KEY, "labresultoffset", "feature"], keep="last")
              .sort_values([pp.KEY, "feature", "labresultoffset"]))

    out = {}
    for s in stays:
        p = pat.loc[s]
        clinical = {k: (None if pd.isna(p[k]) else p[k]) for k in
                    ["age", "admissionweight", "admissionheight", "gender", "unittype", "unitstaytype"]}
        clinical |= {k: int(aps.loc[s, k]) for k in pp.APACHE_FLAGS}
        for df, variables in [(vp, pp.PERIODIC_VITALS), (va, pp.APERIODIC_VITALS)]:
            rows = df[df[pp.KEY] == s]
            clinical |= {v: rows[v].dropna().tolist() for v in variables}
        rows = lab[lab[pp.KEY] == s]
        clinical |= {f: rows.loc[rows["feature"] == f, "labresult"].tolist() for f in pp.LAB_NAMES}
        out[s] = clinical
    return out


def test_round_trip_reproduces_processed_training_rows():
    """Raw readings -> adapter must give exactly the row preprocessing.py wrote for that stay."""
    data = pd.read_csv(pp.OUTPUT_PATH)
    measured = [c for _, cols in inf._measurement_groups().values() for c in cols
                if c not in fa.STATIC_NUMERIC]
    data = data[data[measured].notna().any(axis=1)]  # stays with no vitals/labs are rejected by design
    sample = data.sample(40, random_state=0)
    clinical = _raw_clinical_inputs(sample[pp.KEY].tolist())
    for _, row in sample.iterrows():
        record = fa.build_model_features(clinical[row[pp.KEY]])
        for name in inf.feature_names():
            expected, got = row[name], record[name]
            if pd.isna(expected):
                assert got is None, (row[pp.KEY], name, got)
            else:
                assert got == pytest.approx(float(expected), rel=1e-9, abs=1e-9), (row[pp.KEY], name)
