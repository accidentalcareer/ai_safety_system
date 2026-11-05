"""
Drug Interaction Checker – the checking logic (no Streamlit code here).

How it works, step by step:
1. Clean up each medication name (trim spaces, lower case, map aliases
   such as "albuterol" -> "salbutamol").
2. Sort the names into: blank entries, unknown drugs (not in the sample
   dataset), duplicates, and drugs that can be checked.
3. Compare every pair of checkable drugs against the documented rules
   in data/drug_interaction_rules.py.

Every result produced here is a DOCUMENTED RULE looked up from a table.
Nothing in this module is a machine-learning prediction.

This is an educational demo using synthetic patients. It is not a
clinical decision-support system and does not recommend medication changes.
"""

from dataclasses import dataclass, field
from itertools import combinations

from data.drug_interaction_rules import DRUG_ALIASES, INTERACTION_RULES, OTHER_KNOWN_DRUGS

# Label attached to every result, so the UI can show where it came from.
RESULT_TYPE_RULE = "Documented rule"
RESULT_TYPE_ML = "Machine-learning prediction"  # not produced by this module

# Every drug name that appears in at least one rule.
DRUGS_IN_RULES = {drug for rule in INTERACTION_RULES for drug in rule["drugs"]}

# All drug names the checker recognises.
KNOWN_DRUGS = DRUGS_IN_RULES | OTHER_KNOWN_DRUGS


@dataclass
class Interaction:
    """One detected interaction between two drugs."""

    drug_a: str
    drug_b: str
    concern: str
    explanation: str
    sources: list
    context_labs: list
    context_vitals: list
    result_type: str = RESULT_TYPE_RULE


@dataclass
class CheckResult:
    """Everything the checker found for one list of medications."""

    checked_drugs: list = field(default_factory=list)   # canonical names that were checked
    interactions: list = field(default_factory=list)    # list of Interaction
    unknown_drugs: list = field(default_factory=list)   # names not in the sample dataset
    duplicates: list = field(default_factory=list)      # names listed more than once
    blank_entries: int = 0                              # empty / missing names skipped

    @property
    def has_input(self):
        """True if at least one usable medication name was given."""
        return bool(self.checked_drugs or self.unknown_drugs)


def normalize_name(name):
    """
    Turn a medication name into the standard lower-case generic name.

    Returns None for missing or blank names (None, "", "   ", non-text).
    Unknown names are returned cleaned up but otherwise unchanged.
    """
    if not isinstance(name, str):
        return None
    cleaned = " ".join(name.strip().lower().split())  # also collapses double spaces
    if not cleaned:
        return None
    return DRUG_ALIASES.get(cleaned, cleaned)


def is_known_drug(name):
    """True if the (normalised) name is in the sample dataset."""
    return normalize_name(name) in KNOWN_DRUGS


def find_rule(drug_a, drug_b):
    """Return the rule for a pair of drugs (in any order), or None."""
    pair = {normalize_name(drug_a), normalize_name(drug_b)}
    for rule in INTERACTION_RULES:
        if set(rule["drugs"]) == pair:
            return rule
    return None


def check_interactions(medication_names):
    """
    Check a list of medication names against the sample rules.

    `medication_names` can be any list of strings. None or an empty list
    is accepted and simply gives an empty result.
    """
    result = CheckResult()
    seen = set()

    for raw_name in medication_names or []:
        name = normalize_name(raw_name)
        if name is None:
            result.blank_entries += 1
        elif name in seen:
            if name not in result.duplicates:
                result.duplicates.append(name)
        elif name not in KNOWN_DRUGS:
            seen.add(name)
            result.unknown_drugs.append(raw_name.strip())
        else:
            seen.add(name)
            result.checked_drugs.append(name)

    for drug_a, drug_b in combinations(result.checked_drugs, 2):
        rule = find_rule(drug_a, drug_b)
        if rule:
            result.interactions.append(
                Interaction(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    concern=rule["concern"],
                    explanation=rule["explanation"],
                    sources=rule["sources"],
                    context_labs=rule["context_labs"],
                    context_vitals=rule["context_vitals"],
                )
            )
    return result


def patient_medication_names(patient):
    """
    Get the medication names from a patient record.

    Copes with a missing medication list or entries without a name
    (those come back as None and are counted as blank by the checker).
    """
    return [med.get("name") for med in (patient or {}).get("medications") or []]


def patient_context(interaction, labs, latest_vitals, vital_info):
    """
    Collect the patient's recorded values that are relevant to an interaction.

    labs:          DataFrame from get_lab_results() (may be empty)
    latest_vitals: one row (Series) of the latest vital signs, or None
    vital_info:    the VITAL_SIGNS dictionary (labels and units)

    Returns a list of dicts with "Item", "Value" and "Note". Values are
    shown as recorded; they are NOT interpreted.
    """
    rows = []
    has_labs = labs is not None and not labs.empty
    for test in interaction.context_labs:
        match = labs[labs["Test"] == test] if has_labs else None
        if match is not None and not match.empty:
            lab = match.iloc[0]
            rows.append({
                "Item": test,
                "Value": f"{lab['Value']} {lab['Unit']}",
                "Note": f"Reference range {lab['Reference range']} ({lab['Status']})",
            })
        else:
            rows.append({"Item": test, "Value": "Not recorded", "Note": "No result in this synthetic record"})

    for key in interaction.context_vitals:
        info = vital_info.get(key, {"label": key, "unit": ""})
        if latest_vitals is not None and key in latest_vitals:
            rows.append({
                "Item": f"{info['label']} (latest)",
                "Value": f"{latest_vitals[key]:g} {info['unit']}",
                "Note": f"Typical adult range {info['low']}–{info['high']}",
            })
        else:
            rows.append({"Item": info["label"], "Value": "Not recorded", "Note": "No reading available"})
    return rows


def display_name(drug):
    """'warfarin' -> 'Warfarin' for nicer display."""
    return drug[:1].upper() + drug[1:]
