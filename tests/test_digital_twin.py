"""Unit tests for the Patient Digital Twin integration logic."""

import copy
import math
from datetime import datetime, timedelta

import pytest

from data.synthetic_patients import PATIENTS, generate_vitals, get_patient
from utils import digital_twin as dt
from utils.deterioration import compute_demo_score
from utils.digital_twin import (
    MODULE_DETERIORATION,
    MODULE_INTERACTIONS,
    SOURCE_RULE,
    SOURCE_SIMULATED,
    STATUS_CURRENT,
    STATUS_NEEDS_REASSESSMENT,
    TIMELINE_COLUMNS,
    PatientTwin,
    TwinInputError,
    TwinRegistry,
    timeline_table,
)
from utils.drug_interactions import check_interactions, patient_medication_names


class FakeClock:
    """Deterministic clock: each call is one minute later."""

    def __init__(self):
        self.now = datetime(2026, 10, 13, 9, 0)

    def __call__(self):
        self.now += timedelta(minutes=1)
        return self.now


@pytest.fixture
def twin():
    return PatientTwin("SYN-001", clock=FakeClock())


def _pairs(twin):
    return {frozenset((i.drug_a, i.drug_b)) for i in twin.interactions.interactions}


# --- creation and shared state ----------------------------------------------

@pytest.mark.parametrize("patient_id", [p["patient_id"] for p in PATIENTS])
def test_twin_starts_from_original_record(patient_id):
    twin = PatientTwin(patient_id)
    original = get_patient(patient_id)
    assert twin.medication_names == [m["name"] for m in original["medications"]]
    assert twin.labs == original["labs"]
    assert twin.vitals.equals(generate_vitals(patient_id))
    # Assessments come from the existing modules, unchanged.
    assert twin.interactions == check_interactions(patient_medication_names(original))
    assert twin.deterioration.score == compute_demo_score(generate_vitals(patient_id), original).score
    assert twin.deterioration_status == STATUS_CURRENT
    assert [e.change_type for e in twin.events] == ["Twin created"]


def test_unknown_patient_raises():
    with pytest.raises(KeyError):
        PatientTwin("SYN-999")


def test_edits_never_change_the_shared_patient_records(twin):
    before = copy.deepcopy(PATIENTS)
    twin.add_medication("Ibuprofen")
    twin.remove_medication("Warfarin")
    twin.update_lab("INR", 3.4)
    twin.update_vital("heart_rate", 120)
    assert PATIENTS == before
    assert generate_vitals("SYN-001").iloc[-1]["heart_rate"] != 120


# --- medication workflow ----------------------------------------------------

def test_adding_medication_reruns_checker_and_flags_deterioration(twin):
    score_before = twin.deterioration.score
    event = twin.add_medication("ibuprofen")

    assert "Ibuprofen" in twin.medication_names
    assert frozenset(("warfarin", "ibuprofen")) in _pairs(twin)
    assert frozenset(("lisinopril", "ibuprofen")) in _pairs(twin)
    # The deterioration score has no medication inputs: flagged, not recomputed.
    assert twin.deterioration_status == STATUS_NEEDS_REASSESSMENT
    assert "does not use medication data" in twin.deterioration_note
    assert twin.deterioration.score == score_before

    assert event.change_type == "Medication added"
    assert event.modules_reassessed == [MODULE_INTERACTIONS]
    assert "Warfarin + Ibuprofen" in " ".join(event.reasons)
    assert any(r.startswith(f"{MODULE_DETERIORATION}: not reassessed") for r in event.reasons)
    assert SOURCE_RULE in event.sources and SOURCE_SIMULATED not in event.sources
    assert "Ibuprofen" in event.new_value and "Ibuprofen" not in event.previous_value


def test_removing_medication_removes_matching_rule(twin):
    assert frozenset(("warfarin", "aspirin")) in _pairs(twin)
    event = twin.remove_medication("aspirin")  # case-insensitive
    assert "Aspirin" not in twin.medication_names
    assert not twin.interactions.interactions
    assert "no longer matched: Warfarin + Aspirin" in " ".join(event.reasons)
    assert event.change_type == "Medication removed"


def test_medication_flag_matches_what_the_score_actually_uses(twin):
    """The 'does not use medication data' claim is checked, not just asserted."""
    twin.add_medication("Ibuprofen")
    twin.remove_medication("Warfarin")
    assert compute_demo_score(twin.vitals, twin.record).score == twin.deterioration.score


def test_medication_change_recomputes_if_module_supports_medications(twin, monkeypatch):
    monkeypatch.setattr(dt, "USES_MEDICATION_DATA", True)
    event = twin.add_medication("Ibuprofen")
    assert twin.deterioration_status == STATUS_CURRENT
    assert event.modules_reassessed == [MODULE_INTERACTIONS, MODULE_DETERIORATION]
    assert SOURCE_SIMULATED in event.sources


def test_unknown_medication_is_added_but_reported_as_unchecked(twin):
    event = twin.add_medication("  Unknownium ")
    assert "Unknownium" in twin.medication_names
    assert twin.interactions.unknown_drugs == ["Unknownium"]
    assert "not in the sample dataset" in " ".join(event.reasons)


@pytest.mark.parametrize("name", ["Warfarin", "WARFARIN", "ASA", "acetylsalicylic acid"])
def test_duplicate_medication_is_rejected(twin, name):
    events, meds = len(twin.events), list(twin.medication_names)
    with pytest.raises(TwinInputError, match="already on the simulated medication list"):
        twin.add_medication(name)
    assert twin.medication_names == meds and len(twin.events) == events


@pytest.mark.parametrize("name", [None, "", "   ", 42])
def test_blank_medication_name_is_rejected(twin, name):
    with pytest.raises(TwinInputError):
        twin.add_medication(name)
    assert len(twin.events) == 1


@pytest.mark.parametrize("name", ["Ibuprofen", "", None])
def test_removing_medication_not_on_list_is_rejected(twin, name):
    with pytest.raises(TwinInputError, match="not on the simulated medication list"):
        twin.remove_medication(name)
    assert len(twin.events) == 1


def test_all_medications_can_be_removed(twin):
    for name in list(twin.medication_names):
        twin.remove_medication(name)
    assert twin.medication_names == []
    assert twin.medications_table().empty
    assert not twin.interactions.has_input


# --- vital-sign and lab workflow ----------------------------------------------

def test_vital_change_recomputes_score_and_records_values(twin):
    old = twin.latest_vitals()["resp_rate"]
    before = twin.deterioration.score
    event = twin.update_vital("resp_rate", 30)

    assert twin.latest_vitals()["resp_rate"] == 30
    assert twin.deterioration.score > before
    assert event.previous_value == f"{old:g} /min" and event.new_value == "30 /min"
    assert event.modules_reassessed == [MODULE_DETERIORATION]
    assert SOURCE_SIMULATED in event.sources


def test_vital_change_does_not_claim_to_affect_unrelated_interactions(twin):
    # SYN-001 matches only Warfarin + Aspirin, whose rule lists no vital signs.
    event = twin.update_vital("heart_rate", 120)
    assert MODULE_INTERACTIONS not in event.modules_reassessed
    assert f"{MODULE_INTERACTIONS}: not reassessed" in " ".join(event.reasons)


def test_vital_change_reevaluates_interaction_that_lists_it_as_context():
    twin = PatientTwin("SYN-003")  # Sertraline + Sumatriptan lists heart rate
    pairs_before = _pairs(twin)
    event = twin.update_vital("heart_rate", 115)
    assert MODULE_INTERACTIONS in event.modules_reassessed
    assert "Sertraline + Sumatriptan" in " ".join(event.reasons)
    assert _pairs(twin) == pairs_before  # rule matches never depend on vitals


def test_vital_change_after_medication_change_clears_reassessment_flag(twin):
    twin.add_medication("Ibuprofen")
    twin.update_vital("systolic_bp", 165)
    assert twin.deterioration_status == STATUS_CURRENT
    # Lisinopril + Ibuprofen lists systolic BP as context.
    assert MODULE_INTERACTIONS in twin.events[-1].modules_reassessed


def test_marking_vital_missing_is_handled(twin):
    event = twin.update_vital("spo2", None)
    assert math.isnan(twin.latest_vitals()["spo2"])
    assert event.new_value == "missing"
    assert any("missing at the latest observation time" in m for m in twin.deterioration.missing)


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("spo2", 150, "outside the data-entry limits"),
        ("heart_rate", -5, "outside the data-entry limits"),
        ("heart_rate", "fast", "not a number"),
        ("heart_rate", float("inf"), "finite"),
        ("heart_rate", True, "not a number"),
        ("blood_sugar", 5, "not a supported vital sign"),
    ],
)
def test_invalid_vital_inputs_are_rejected(twin, key, value, message):
    vitals = twin.vitals.copy()
    with pytest.raises(TwinInputError, match=message):
        twin.update_vital(key, value)
    assert twin.vitals.equals(vitals) and len(twin.events) == 1


def test_unchanged_vital_value_is_not_logged(twin):
    current = twin.latest_vitals()["heart_rate"]
    with pytest.raises(TwinInputError, match="nothing changed"):
        twin.update_vital("heart_rate", current)
    assert len(twin.events) == 1


def test_lab_used_by_score_recomputes_it(twin):
    before = twin.deterioration.score
    event = twin.update_lab("Potassium", 6.0)
    assert twin.lab_value("Potassium") == 6.0
    assert twin.deterioration.score > before
    assert MODULE_DETERIORATION in event.modules_reassessed
    assert event.previous_value == "4.9 mmol/L" and event.new_value == "6 mmol/L"


def test_lab_not_used_by_score_only_refreshes_interaction_context(twin):
    before = twin.deterioration.score
    event = twin.update_lab("INR", 3.8)  # Warfarin + Aspirin lists INR as context
    assert twin.deterioration.score == before
    assert event.modules_reassessed == [MODULE_INTERACTIONS]
    assert f"{MODULE_DETERIORATION}: not reassessed" in " ".join(event.reasons)
    assert twin.lab_table().set_index("Test").loc["INR", "Value"] == 3.8


def test_lab_with_no_related_module_reassesses_nothing(twin):
    event = twin.update_lab("HbA1c", 8.0)
    assert event.modules_reassessed == []


@pytest.mark.parametrize(
    "test, value, message",
    [
        ("C-reactive protein", 5, "not recorded"),   # not in SYN-001's record
        ("Potassium", "high", "not a number"),
        ("Potassium", None, "not a number"),
        ("Potassium", float("nan"), "cannot be blank"),
        ("Potassium", -1, "cannot be negative"),
        ("Potassium", 4.9, "nothing changed"),
    ],
)
def test_invalid_lab_inputs_are_rejected(twin, test, value, message):
    labs = list(twin.labs)
    with pytest.raises(TwinInputError, match=message):
        twin.update_lab(test, value)
    assert twin.labs == labs and len(twin.events) == 1


# --- reassessment, reset, isolation, timeline --------------------------------------

def test_manual_reassessment_clears_flag_and_explains_unchanged_score(twin):
    twin.add_medication("Ibuprofen")
    score = twin.deterioration.score
    event = twin.reassess()
    assert twin.deterioration_status == STATUS_CURRENT
    assert twin.deterioration.score == score
    assert event.modules_reassessed == [MODULE_INTERACTIONS, MODULE_DETERIORATION]
    assert "no medication inputs" in " ".join(event.reasons)


def test_reset_restores_original_values_and_keeps_history(twin):
    original = PatientTwin("SYN-001")
    twin.add_medication("Ibuprofen")
    twin.update_vital("spo2", 88)
    twin.update_lab("Potassium", 6.2)

    event = twin.reset()

    assert twin.medication_names == original.medication_names
    assert twin.labs == original.labs
    assert twin.vitals.equals(original.vitals)
    assert twin.deterioration.score == original.deterioration.score
    assert _pairs(twin) == _pairs(original)
    assert twin.deterioration_status == STATUS_CURRENT
    assert event.change_type == "Reset"
    assert event.previous_value.startswith("3 simulated change(s)")
    assert len(twin.events) == 5  # created + 3 changes + reset


def test_second_reset_counts_only_changes_since_last_reset(twin):
    twin.add_medication("Ibuprofen")
    twin.reset()
    twin.update_vital("heart_rate", 110)
    assert twin.reset().previous_value.startswith("1 simulated change(s)")


def test_registry_keeps_patients_separate():
    registry = TwinRegistry(clock=FakeClock())
    a, b = registry.get("SYN-001"), registry.get("SYN-002")
    assert registry.get("SYN-001") is a  # same twin on every call

    a.add_medication("Ibuprofen")
    a.update_vital("heart_rate", 130)

    assert "Ibuprofen" not in b.medication_names
    assert b.vitals.equals(generate_vitals("SYN-002"))
    assert len(b.events) == 1
    assert {e.patient_id for e in a.events} == {"SYN-001"}
    assert "SYN-003" not in registry

    a.reset()
    assert len(b.events) == 1


def test_events_are_numbered_timestamped_and_complete(twin):
    twin.add_medication("Ibuprofen")
    twin.update_vital("heart_rate", 120)
    assert [e.event_id for e in twin.events] == [1, 2, 3]
    times = [e.timestamp for e in twin.events]
    assert times == sorted(times) and len(set(times)) == 3
    for event in twin.events:
        assert event.patient_id == "SYN-001"
        assert event.status and event.sources and event.reasons


def test_timeline_table_has_required_columns_newest_first(twin):
    twin.add_medication("Ibuprofen")
    twin.update_vital("heart_rate", 120)
    table = timeline_table(twin.events)
    assert list(table.columns) == TIMELINE_COLUMNS
    assert list(table["#"]) == [3, 2, 1]
    row = table.iloc[0]
    assert row["Change"] == "Vital sign changed" and row["New"] == "120 bpm"
    assert row["Modules reassessed"] == MODULE_DETERIORATION
    assert "Simulated demonstration score" in row["Output source"]
    assert "Demo score" in row["Resulting status"]


def test_no_output_is_labelled_as_an_evaluated_model(twin):
    twin.add_medication("Ibuprofen")
    twin.update_vital("heart_rate", 120)
    twin.reassess()
    for event in twin.events:
        assert not any("model" in source.lower() for source in event.sources)
