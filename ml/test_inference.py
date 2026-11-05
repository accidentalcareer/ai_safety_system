"""Tests for ml/inference.py.

Run:  .venv\\Scripts\\python.exe -m pytest ml/test_inference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference as inf  # noqa: E402

DATA_PATH = inf.ML_DIR / "data" / "processed" / "mortality_training.csv"
SPLIT_PATH = inf.ML_DIR / "results" / "split.csv"
EXPECTED_KEYS = {"risk_score", "risk_category", "category_description", "above_threshold",
                 "decision_threshold", "score_type", "model_name", "model_version",
                 "n_features_not_recorded", "features_not_recorded", "disclaimer"}


@pytest.fixture(scope="module")
def test_rows() -> pd.DataFrame:
    """Held-out test-split stays from the processed dataset (real, schema-correct records)."""
    data = pd.read_csv(DATA_PATH)
    split = pd.read_csv(SPLIT_PATH)
    test_ids = split.loc[split["split"] == "test", "patientunitstayid"]
    return data[data["patientunitstayid"].isin(test_ids)].reset_index(drop=True)


@pytest.fixture(scope="module")
def scorable_rows(test_rows) -> pd.DataFrame:
    """Test-split stays with at least one vital or lab value (the rest are rejected by design)."""
    measured = [c for _, cols in inf._measurement_groups().values() for c in cols
                if c not in inf.STATIC_NUMERIC]
    return test_rows[test_rows[measured].notna().any(axis=1)].reset_index(drop=True)


def as_record(row: pd.Series) -> dict:
    """Feature dict with blanks as None, as a caller would build it."""
    return {f: (None if pd.isna(row[f]) else getattr(row[f], "item", lambda: row[f])())
            for f in inf.feature_names()}


@pytest.fixture
def valid_record(test_rows) -> dict:
    return as_record(test_rows.iloc[0])


# --------------------------------------------------------------------------- model loading

def test_model_file_loads():
    model, meta = inf.load_model()
    assert hasattr(model, "predict_proba")
    assert len(meta["feature_names"]) == inf.EXPECTED_N_FEATURES == 126
    assert list(model.feature_names_in_) == meta["feature_names"]
    assert 0 < meta["decision_threshold"] < 1
    assert inf.feature_names() == meta["feature_names"]


# --------------------------------------------------------------------------- valid input

def test_valid_input_produces_result(valid_record):
    result = inf.predict_mortality(valid_record)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_category"] in {inf.CATEGORY_LOWER, inf.CATEGORY_ELEVATED}


def test_result_structure(valid_record):
    result = inf.predict_mortality(valid_record)
    assert set(result) == EXPECTED_KEYS
    meta = json.loads(inf.METADATA_PATH.read_text())
    assert result["model_name"] == meta["model_name"] == "Logistic Regression"
    assert result["decision_threshold"] == meta["decision_threshold"]
    assert result["above_threshold"] == (result["risk_category"] == inf.CATEGORY_ELEVATED)
    assert "not a probability" in result["score_type"]
    assert "%" not in result["category_description"]
    assert result["n_features_not_recorded"] == len(result["features_not_recorded"])
    json.dumps(result)  # plain JSON-serialisable types only


def test_repeated_input_is_deterministic(valid_record):
    results = [inf.predict_mortality(dict(valid_record)) for _ in range(5)]
    assert all(r == results[0] for r in results)


def test_score_matches_saved_pipeline(scorable_rows):
    """The wrapper must add nothing to the saved pipeline's own output."""
    model, meta = inf.load_model()
    direct = model.predict_proba(scorable_rows[meta["feature_names"]])[:, 1]
    wrapped = [inf.predict_mortality(as_record(r))["risk_score"] for _, r in scorable_rows.iterrows()]
    np.testing.assert_allclose(wrapped, direct, rtol=0, atol=1e-12)


def test_category_follows_decision_threshold(scorable_rows):
    threshold = inf.load_model()[1]["decision_threshold"]
    categories = set()
    for _, row in scorable_rows.iterrows():
        r = inf.predict_mortality(as_record(row))
        assert r["risk_category"] == (inf.CATEGORY_ELEVATED if r["risk_score"] >= threshold
                                      else inf.CATEGORY_LOWER)
        categories.add(r["risk_category"])
    assert categories == {inf.CATEGORY_LOWER, inf.CATEGORY_ELEVATED}
    assert inf.risk_category(threshold, threshold) == inf.CATEGORY_ELEVATED


def test_accepts_series_and_one_row_dataframe(test_rows, valid_record):
    expected = inf.predict_mortality(valid_record)["risk_score"]
    row = test_rows.iloc[0][inf.feature_names()]
    assert inf.predict_mortality(row)["risk_score"] == expected
    assert inf.predict_mortality(row.to_frame().T)["risk_score"] == expected


# --------------------------------------------------------------------------- invalid input

def test_missing_required_feature_is_detected(valid_record):
    for dropped in ["age", "lactate_max", "unittype_micu"]:
        record = {k: v for k, v in valid_record.items() if k != dropped}
        with pytest.raises(inf.MortalityInputError, match=dropped):
            inf.predict_mortality(record)


def test_many_missing_features_reported_together(valid_record):
    record = {k: valid_record[k] for k in list(valid_record)[:10]}
    with pytest.raises(inf.MortalityInputError) as exc:
        inf.predict_mortality(record)
    assert "116 required feature(s) missing" in str(exc.value)


@pytest.mark.parametrize("empty", [{}, pd.Series(dtype=float), pd.DataFrame(), None, [], "patient"])
def test_empty_or_wrong_type_input_rejected(empty):
    with pytest.raises(inf.MortalityInputError):
        inf.predict_mortality(empty)


def test_record_with_every_measurement_blank_is_rejected(valid_record):
    """All keys present but no vitals/labs: must not yield a score built purely on imputation."""
    record = dict(valid_record)
    for indicator, (_, cols) in inf._measurement_groups().items():
        record[indicator] = 1
        for c in cols:
            record[c] = None
    with pytest.raises(inf.MortalityInputError, match="No vital signs or laboratory values"):
        inf.predict_mortality(record)


def test_unknown_feature_rejected(valid_record):
    with pytest.raises(inf.MortalityInputError, match="heart_rate"):
        inf.predict_mortality({**valid_record, "heart_rate": 80})


@pytest.mark.parametrize("value", ["72", np.inf, [1, 2]])
def test_non_numeric_value_rejected(valid_record, value):
    with pytest.raises(inf.MortalityInputError, match="heartrate_mean"):
        inf.predict_mortality({**valid_record, "heartrate_mean": value})


def test_blank_value_must_be_flagged_missing(valid_record):
    record = {**valid_record, "heartrate_mean": None, "heartrate_missing": 0}
    with pytest.raises(inf.MortalityInputError, match="heartrate_missing"):
        inf.predict_mortality(record)


def test_missing_flag_with_values_rejected(valid_record):
    with pytest.raises(inf.MortalityInputError, match="sodium_missing=1"):
        inf.predict_mortality({**valid_record, "sodium_missing": 1})


def test_out_of_range_value_rejected(valid_record):
    with pytest.raises(inf.MortalityInputError, match="plausible range"):
        inf.predict_mortality({**valid_record, "age": 150})


def test_binary_and_one_hot_rules(valid_record):
    with pytest.raises(inf.MortalityInputError, match="apache_vent"):
        inf.predict_mortality({**valid_record, "apache_vent": 2})
    gender_off = {**valid_record, "gender_female": 0, "gender_male": 0, "gender_missing": 0}
    with pytest.raises(inf.MortalityInputError, match="exactly one"):
        inf.predict_mortality(gender_off)


def test_all_training_records_with_measurements_validate():
    """Every record preprocessing.py produced must pass validation unless it has no vitals/labs."""
    data = pd.read_csv(DATA_PATH)
    rejected = 0
    for _, row in data.iterrows():
        try:
            inf.validate_features(as_record(row))
        except inf.MortalityInputError as e:
            assert len(e.errors) == 1 and "No vital signs" in e.errors[0], e.errors
            rejected += 1
    assert rejected == 6  # the 6 stays with no vital or lab values in the 24h window
