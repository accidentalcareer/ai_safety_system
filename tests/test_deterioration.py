"""
Tests for the Deterioration Prediction demo score. Run with:  python -m pytest

The tests check BEHAVIOUR (e.g. "a reading further outside its range adds
more points", "missing data is reported") rather than clinical claims.
Ranges come from the app's own illustrative VITAL_SIGNS table, never from
numbers hard-coded here.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from data.synthetic_patients import PATIENTS, VITAL_SIGNS, generate_vitals, get_patient
from utils.deterioration import (
    DEMO_LABS,
    MAX_PART_POINTS,
    PLAUSIBLE_LIMITS,
    RESULT_TYPE_SIMULATED,
    SCORE_VITALS,
    clean_vitals,
    compute_demo_score,
    contributions_table,
    lab_inputs,
    labeled_data_check,
    range_deviation,
    score_history,
    vital_features,
)

START = datetime(2026, 1, 1, 8, 0)


def _midpoint(key):
    info = VITAL_SIGNS[key]
    return (info["low"] + info["high"]) / 2


def _vitals(n=13, hours_apart=2, **overrides):
    """A table of steady mid-range readings; override any column with a list."""
    data = {"timestamp": [START + timedelta(hours=i * hours_apart) for i in range(n)]}
    for key in SCORE_VITALS:
        data[key] = overrides.get(key, [_midpoint(key)] * n)
    return pd.DataFrame(data)


def _points(result, key):
    return next(c.points for c in result.contributions if c.key == key)


# --- Mode decision ----------------------------------------------------------

def test_synthetic_data_has_no_labels_so_no_model_is_trained():
    available, reason = labeled_data_check()
    assert available is False
    assert "outcome label" in reason


def test_label_check_requires_both_outcome_classes():
    assert labeled_data_check([{"deterioration_outcome": 0}, {"deterioration_outcome": 0}])[0] is False
    assert labeled_data_check([{"deterioration_outcome": 0}, {"deterioration_outcome": 1}])[0] is True


def test_result_is_labelled_as_simulated():
    result = compute_demo_score(_vitals(), None)
    assert result.result_type == RESULT_TYPE_SIMULATED
    assert "Simulated" in RESULT_TYPE_SIMULATED


# --- Preprocessing: range deviation -----------------------------------------

def test_range_deviation():
    assert range_deviation(15, 10, 20) == 0
    assert range_deviation(10, 10, 20) == 0          # boundaries count as inside
    assert range_deviation(25, 10, 20) == pytest.approx(0.5)
    assert range_deviation(5, 10, 20) == pytest.approx(0.5)


def test_range_deviation_rejects_empty_range():
    with pytest.raises(ValueError):
        range_deviation(5, 10, 10)


# --- Normal inputs ----------------------------------------------------------

def test_all_mid_range_vitals_give_zero_vital_points():
    result = compute_demo_score(_vitals(), None, include_labs=False)
    assert result.score == 0
    assert result.items_used == result.items_expected == len(SCORE_VITALS)
    assert result.invalid == [] and result.missing == []


@pytest.mark.parametrize("patient", PATIENTS, ids=lambda p: p["patient_id"])
def test_every_synthetic_patient_scores_without_errors(patient):
    result = compute_demo_score(generate_vitals(patient["patient_id"]), patient)
    assert result.score >= 0
    assert result.invalid == []
    assert 0 < result.coverage <= 1
    assert result.score == pytest.approx(sum(c.points for c in result.contributions))


def test_drifting_synthetic_patient_scores_above_steady_ones():
    """SYN-002's profile is built to drift; SYN-001 and SYN-003 are steady."""
    scores = {p["patient_id"]: compute_demo_score(generate_vitals(p["patient_id"]), p).score for p in PATIENTS}
    assert scores["SYN-002"] > scores["SYN-001"]
    assert scores["SYN-002"] > scores["SYN-003"]


def test_contributions_are_sorted_and_tabulated():
    pid = "SYN-002"
    result = compute_demo_score(generate_vitals(pid), get_patient(pid))
    points = [c.points for c in result.contributions]
    assert points == sorted(points, reverse=True)
    table = contributions_table(result)
    assert len(table) == len(result.contributions)
    assert {"Factor", "Points", "Outside range", "Worsening (24 h)"} <= set(table.columns)


# --- Changing vital signs ---------------------------------------------------

@pytest.mark.parametrize("key", SCORE_VITALS)
def test_further_outside_range_gives_more_points(key):
    info = VITAL_SIGNS[key]
    width = info["high"] - info["low"]
    # Go above the range, or below it if values above are impossible (e.g. SpO2 > 100 %).
    above_possible = info["high"] + 0.5 * width <= PLAUSIBLE_LIMITS[key][1]
    scores = []
    for steps_outside in [0, 0.25, 0.5]:
        if above_possible:
            latest = info["high"] + steps_outside * width
        else:
            latest = info["low"] - steps_outside * width
        values = [_midpoint(key)] * 12 + [latest]
        scores.append(compute_demo_score(_vitals(**{key: values}), None, include_labs=False).score)
    assert scores[0] <= scores[1] <= scores[2]
    assert scores[2] > scores[0]


def test_below_range_also_adds_points():
    low = VITAL_SIGNS["spo2"]["low"]
    values = [_midpoint("spo2")] * 12 + [low - 2]
    assert _points(compute_demo_score(_vitals(spo2=values), None, include_labs=False), "spo2") > 0


def test_worsening_trend_adds_points_but_improvement_does_not():
    info = VITAL_SIGNS["resp_rate"]
    width = info["high"] - info["low"]
    far, near = info["high"] + 0.8 * width, info["high"] + 0.2 * width

    worsening = compute_demo_score(_vitals(resp_rate=[near] * 12 + [far]), None, include_labs=False)
    steady = compute_demo_score(_vitals(resp_rate=[far] * 13), None, include_labs=False)
    improving = compute_demo_score(_vitals(resp_rate=[far] * 12 + [near]), None, include_labs=False)

    contrib = next(c for c in worsening.contributions if c.key == "resp_rate")
    assert contrib.worsening > 0
    assert worsening.score > steady.score
    assert next(c for c in improving.contributions if c.key == "resp_rate").worsening == 0


def test_trend_only_looks_at_last_24_hours():
    """An old out-of-range value more than 24 h ago is not used for the trend."""
    high = VITAL_SIGNS["heart_rate"]["high"]
    values = [high * 1.5] + [_midpoint("heart_rate")] * 20 + [high * 1.5]
    feats = vital_features(clean_vitals(_vitals(n=22, heart_rate=values))[0], "heart_rate")
    assert feats["earlier"] == _midpoint("heart_rate")


def test_each_part_is_capped():
    huge = PLAUSIBLE_LIMITS["heart_rate"][1]
    result = compute_demo_score(_vitals(heart_rate=[_midpoint("heart_rate")] * 12 + [huge]), None, include_labs=False)
    contrib = next(c for c in result.contributions if c.key == "heart_rate")
    assert contrib.deviation == MAX_PART_POINTS
    assert contrib.points <= 2 * MAX_PART_POINTS


def test_score_history_follows_the_readings():
    info = VITAL_SIGNS["spo2"]
    values = [_midpoint("spo2")] * 6 + [info["low"] - 3] * 7
    history = score_history(_vitals(spo2=values))
    assert len(history) == 13
    assert history["score"].iloc[0] == 0
    assert history["score"].iloc[-1] > 0
    assert history["timestamp"].is_monotonic_increasing


# --- Missing values ---------------------------------------------------------

def test_missing_vital_column_is_reported():
    df = _vitals().drop(columns=["spo2"])
    result = compute_demo_score(df, None, include_labs=False)
    assert result.items_used == len(SCORE_VITALS) - 1
    assert any("SpO" in note for note in result.missing)


def test_vital_with_all_nan_is_reported():
    result = compute_demo_score(_vitals(heart_rate=[np.nan] * 13), None, include_labs=False)
    assert "heart_rate" not in {c.key for c in result.contributions}
    assert any("Heart rate" in note for note in result.missing)


def test_missing_latest_reading_uses_earlier_one_and_warns():
    values = [_midpoint("resp_rate")] * 12 + [np.nan]
    result = compute_demo_score(_vitals(resp_rate=values), None, include_labs=False)
    assert "resp_rate" in {c.key for c in result.contributions}
    assert any("missing at the latest observation" in note for note in result.missing)


def test_single_reading_has_no_trend():
    result = compute_demo_score(_vitals(n=1), None, include_labs=False)
    assert all(c.worsening == 0 for c in result.contributions)
    assert any("trend not assessed" in note for note in result.missing)


def test_empty_table_gives_no_score_and_warnings():
    empty = _vitals().iloc[0:0]
    result = compute_demo_score(empty, None)
    assert result.contributions == [] and result.score == 0
    assert result.coverage == 0
    assert any("No vital-sign readings" in note for note in result.missing)


def test_missing_labs_are_reported():
    result = compute_demo_score(_vitals(), {"labs": []})
    for test in DEMO_LABS:
        assert any(note.startswith(test) for note in result.missing)
    # Patient without a labs key / no patient at all must also work.
    assert compute_demo_score(_vitals(), {}).items_used == len(SCORE_VITALS)
    assert compute_demo_score(_vitals(), None).items_used == len(SCORE_VITALS)


def test_crp_not_recorded_for_patient_a():
    result = compute_demo_score(generate_vitals("SYN-001"), get_patient("SYN-001"))
    assert any("C-reactive protein" in note for note in result.missing)


# --- Invalid inputs ---------------------------------------------------------

@pytest.mark.parametrize("bad", [None, [], "vitals", {"timestamp": []}])
def test_non_dataframe_input_raises_type_error(bad):
    with pytest.raises(TypeError):
        compute_demo_score(bad, None)


def test_missing_timestamp_column_raises_value_error():
    with pytest.raises(ValueError):
        compute_demo_score(_vitals().drop(columns=["timestamp"]), None)


def test_non_numeric_values_are_ignored_and_reported():
    df = _vitals()
    df["heart_rate"] = df["heart_rate"].astype(object)
    df.loc[12, "heart_rate"] = "abc"
    cleaned, invalid = clean_vitals(df)
    assert np.isnan(cleaned.loc[12, "heart_rate"])
    assert any("non-numeric" in note for note in invalid)


@pytest.mark.parametrize("key, value", [("spo2", 150), ("heart_rate", -5), ("temperature", 98.6)])
def test_impossible_values_are_ignored_and_reported(key, value):
    values = [_midpoint(key)] * 12 + [value]
    result = compute_demo_score(_vitals(**{key: values}), None, include_labs=False)
    assert any("data-entry limits" in note for note in result.invalid)
    # The impossible value itself is not scored: the earlier mid-range reading is used.
    assert _points(result, key) == 0


def test_bad_timestamps_are_dropped_and_rows_sorted():
    df = _vitals(n=3)
    df["timestamp"] = df["timestamp"].astype(object)
    df.loc[1, "timestamp"] = "not a time"
    df = df.iloc[::-1]  # reverse order
    cleaned, invalid = clean_vitals(df)
    assert len(cleaned) == 2
    assert cleaned["timestamp"].is_monotonic_increasing
    assert any("valid time" in note for note in invalid)


def test_clean_vitals_does_not_modify_input():
    df = _vitals()
    df.loc[12, "spo2"] = 150
    before = df.copy()
    clean_vitals(df)
    assert df.equals(before)


def test_invalid_lab_entries_are_reported():
    patient = {"labs": [
        ("Lactate", "high", "mmol/L", 0.5, 2.0),      # non-numeric value
        ("Sodium", 140, "mmol/L", 145, 135),          # reversed range
        ("Potassium", 4.0, "mmol/L"),                 # wrong shape
        ("Creatinine", 1.0, "mg/dL", 0.6, 1.1),       # valid
    ]}
    labs, invalid = lab_inputs(patient)
    assert list(labs) == ["Creatinine"]
    assert len(invalid) == 3


def test_labs_outside_range_add_points():
    patient = {"labs": [("Lactate", 2.0 + 1.5, "mmol/L", 0.5, 2.0)]}
    result = compute_demo_score(_vitals(), patient)
    assert _points(result, "Lactate") == pytest.approx(1.0)  # (3.5 - 2.0) / 1.5 = 1 range-width
