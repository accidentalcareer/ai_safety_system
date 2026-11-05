"""Tests for the Drug Interaction Checker logic. Run with:  python -m pytest"""

import pandas as pd
import pytest

from data.drug_interaction_rules import INTERACTION_RULES
from data.synthetic_patients import VITAL_SIGNS, get_lab_results, get_patient
from utils.drug_interactions import (
    KNOWN_DRUGS,
    RESULT_TYPE_RULE,
    check_interactions,
    find_rule,
    normalize_name,
    patient_context,
    patient_medication_names,
)


def _pairs(result):
    return {frozenset((i.drug_a, i.drug_b)) for i in result.interactions}


# --- The dataset itself ----------------------------------------------------

def test_every_rule_is_documented():
    for rule in INTERACTION_RULES:
        assert len(rule["drugs"]) == 2 and rule["drugs"][0] != rule["drugs"][1]
        assert rule["concern"] and rule["explanation"]
        assert rule["sources"], f"rule {rule['drugs']} has no source"
        for source in rule["sources"]:
            assert source["title"] and source["section"]
            assert source["url"].startswith("https://")


def test_no_duplicate_rules():
    pairs = [frozenset(rule["drugs"]) for rule in INTERACTION_RULES]
    assert len(pairs) == len(set(pairs))


# --- Known sample interactions ----------------------------------------------

@pytest.mark.parametrize(
    "drug_a, drug_b, concern_word",
    [
        ("Warfarin", "Aspirin", "bleeding"),
        ("Sertraline", "Sumatriptan", "serotonin"),
        ("Sertraline", "Ibuprofen", "bleeding"),
        ("Salbutamol", "Prednisolone", "potassium"),
        ("Atorvastatin", "Clarithromycin", "muscle"),
    ],
)
def test_known_interactions_are_detected(drug_a, drug_b, concern_word):
    result = check_interactions([drug_a, drug_b])
    assert len(result.interactions) == 1
    interaction = result.interactions[0]
    assert concern_word in interaction.concern.lower()
    assert interaction.result_type == RESULT_TYPE_RULE
    assert interaction.sources


def test_order_and_case_do_not_matter():
    assert find_rule("ASPIRIN", " warfarin ") is find_rule("Warfarin", "Aspirin") is not None


@pytest.mark.parametrize(
    "patient_id, expected_pairs",
    [
        ("SYN-001", {frozenset({"warfarin", "aspirin"})}),
        ("SYN-002", {frozenset({"salbutamol", "prednisolone"})}),
        ("SYN-003", {frozenset({"sertraline", "sumatriptan"}), frozenset({"sertraline", "ibuprofen"})}),
    ],
)
def test_synthetic_patients(patient_id, expected_pairs):
    result = check_interactions(patient_medication_names(get_patient(patient_id)))
    assert _pairs(result) == expected_pairs
    assert result.unknown_drugs == [] and result.duplicates == []


# --- Non-interacting pairs --------------------------------------------------

@pytest.mark.parametrize("pair", [("Metformin", "Lisinopril"), ("Omeprazole", "Ceftriaxone"), ("Levothyroxine", "Sumatriptan")])
def test_pairs_without_rule_give_no_interaction(pair):
    result = check_interactions(list(pair))
    assert result.interactions == []
    assert sorted(result.checked_drugs) == sorted(p.lower() for p in pair)


# --- Unknown names, aliases, duplicates -------------------------------------

def test_unknown_medication_is_reported_not_checked():
    result = check_interactions(["Warfarin", "Unknownium", "Aspirin"])
    assert result.unknown_drugs == ["Unknownium"]
    assert "unknownium" not in result.checked_drugs
    assert len(result.interactions) == 1  # warfarin + aspirin still found


def test_only_unknown_medications():
    result = check_interactions(["Foo", "Bar"])
    assert result.interactions == [] and result.checked_drugs == []
    assert result.unknown_drugs == ["Foo", "Bar"]
    assert result.has_input


def test_aliases_map_to_generic_names():
    assert normalize_name("Albuterol") == "salbutamol"
    assert normalize_name("  Acetylsalicylic   Acid ") == "aspirin"
    result = check_interactions(["albuterol", "prednisolone"])
    assert len(result.interactions) == 1


def test_duplicates_are_reported_and_checked_once():
    result = check_interactions(["Warfarin", "warfarin ", "Aspirin", "ASA"])
    assert result.duplicates == ["warfarin", "aspirin"]  # "ASA" is an alias of aspirin
    assert result.checked_drugs == ["warfarin", "aspirin"]
    assert len(result.interactions) == 1


# --- Missing inputs ---------------------------------------------------------

@pytest.mark.parametrize("empty_input", [None, [], ["", "   "], [None], [None, 42]])
def test_missing_inputs(empty_input):
    result = check_interactions(empty_input)
    assert result.interactions == []
    assert not result.has_input


def test_single_medication_has_nothing_to_compare():
    result = check_interactions(["Warfarin"])
    assert result.checked_drugs == ["warfarin"] and result.interactions == []


def test_blank_entries_are_counted():
    result = check_interactions(["Warfarin", "", None, "Aspirin"])
    assert result.blank_entries == 2
    assert len(result.interactions) == 1


def test_patient_without_medication_list():
    assert patient_medication_names({"patient_id": "X"}) == []
    assert patient_medication_names({"medications": None}) == []
    assert patient_medication_names(None) == []
    assert patient_medication_names({"medications": [{"dose": "5 mg"}]}) == [None]


# --- Patient context --------------------------------------------------------

def test_patient_context_shows_recorded_and_missing_values():
    interaction = check_interactions(["Warfarin", "Aspirin"]).interactions[0]
    labs = get_lab_results("SYN-001")
    vitals = pd.Series({"spo2": 97})
    rows = {row["Item"]: row for row in patient_context(interaction, labs, vitals, VITAL_SIGNS)}
    assert rows["INR"]["Value"] == "2.6 ratio"

    # CK is not recorded for any synthetic patient -> shown as missing.
    statin = check_interactions(["Atorvastatin", "Clarithromycin"]).interactions[0]
    rows = patient_context(statin, labs, vitals, VITAL_SIGNS)
    assert rows[0]["Value"] == "Not recorded"


def test_patient_context_with_no_data_at_all():
    interaction = check_interactions(["Salbutamol", "Prednisolone"]).interactions[0]
    rows = patient_context(interaction, pd.DataFrame(), None, VITAL_SIGNS)
    assert [row["Value"] for row in rows] == ["Not recorded", "Not recorded"]


def test_all_patient_medications_are_known_drugs():
    """Every drug in the synthetic data should be recognised (not 'unknown')."""
    for patient_id in ["SYN-001", "SYN-002", "SYN-003"]:
        for name in patient_medication_names(get_patient(patient_id)):
            assert normalize_name(name) in KNOWN_DRUGS
