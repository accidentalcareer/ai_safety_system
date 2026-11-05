"""Basic checks for the synthetic data module. Run with:  python -m pytest"""

import pytest

from data.synthetic_patients import (
    VITAL_SIGNS,
    classify_value,
    generate_vitals,
    get_lab_results,
    get_medications,
    get_patient,
    list_patients,
)


def test_three_clearly_labelled_synthetic_patients():
    patients = list_patients()
    assert len(patients) == 3
    assert patients["patient_id"].is_unique
    assert all(pid.startswith("SYN-") for pid in patients["patient_id"])
    assert all("Synthetic" in name for name in patients["display_name"])


def test_unknown_patient_raises():
    with pytest.raises(KeyError):
        get_patient("SYN-999")


@pytest.mark.parametrize("patient_id", ["SYN-001", "SYN-002", "SYN-003"])
def test_vitals_shape_and_values(patient_id):
    vitals = generate_vitals(patient_id)
    assert len(vitals) == 37  # 72 hours, every 2 hours, inclusive
    for key in VITAL_SIGNS:
        assert key in vitals.columns
        assert vitals[key].notna().all()
    assert vitals["timestamp"].is_monotonic_increasing
    assert vitals["spo2"].between(70, 100).all()
    assert (vitals["systolic_bp"] > vitals["diastolic_bp"]).all()


def test_vitals_are_reproducible():
    assert generate_vitals("SYN-002").equals(generate_vitals("SYN-002"))


@pytest.mark.parametrize("patient_id", ["SYN-001", "SYN-002", "SYN-003"])
def test_medications_and_labs_present(patient_id):
    assert len(get_medications(patient_id)) >= 3
    labs = get_lab_results(patient_id)
    assert set(labs["Status"]) <= {"Low", "Normal", "High"}


def test_classify_value():
    assert classify_value(50, 60, 100) == "Low"
    assert classify_value(80, 60, 100) == "Normal"
    assert classify_value(100, 60, 100) == "Normal"
    assert classify_value(120, 60, 100) == "High"
