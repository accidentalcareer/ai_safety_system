"""
Drug Interaction Checker page – Medication Safety / Drug Interaction Analysis.

Checks the selected synthetic patient's medication list against a small,
documented sample of interaction rules (data/drug_interaction_rules.py)
and explains every match in plain language with its source.

All results on this page are DOCUMENTED RULES from a lookup table, not
machine-learning predictions. Nothing here recommends a medication change.
The rules carry no severity grade (by design), so none is shown.
"""

import pandas as pd
import streamlit as st

from data.drug_interaction_rules import INTERACTION_RULES
from data.synthetic_patients import VITAL_SIGNS, get_lab_results, get_medications, get_patient
from ui.components import (
    badge_row,
    card_container,
    fact_rows,
    label,
    metric_card_html,
    metric_grid,
    page_header,
    section_header,
    summary_block,
    tile_grid,
    tile_html,
)
from utils.drug_interactions import (
    KNOWN_DRUGS,
    RESULT_TYPE_RULE,
    check_interactions,
    display_name,
    is_known_drug,
    normalize_name,
    patient_context,
    patient_medication_names,
)
from views.dashboard import load_vitals

RULE_BADGE = RESULT_TYPE_RULE
SEVERITY_NOT_GRADED = "Severity not graded"


def render(patient_id):
    """Draw the whole Drug Interaction Checker page for one patient."""
    patient = get_patient(patient_id)

    page_header(
        "Drug Interaction Analysis",
        subtitle=(
            f"{patient['display_name']} · Patient ID **{patient['patient_id']}** · Synthetic demonstration: "
            "the medication list is compared with a small teaching sample of documented interaction rules."
        ),
        eyebrow="Medication safety",
        badges=[
            (RESULT_TYPE_RULE, "info"),
            ("Synthetic data", "neutral"),
            (f"{len(INTERACTION_RULES)} sample rules · {len(KNOWN_DRUGS)} recognised drugs", "neutral"),
        ],
    )
    with st.container(key="iss_notice_di"):
        st.warning(
            "**Not a clinical tool.** This checker uses a small teaching sample of interaction rules. It is "
            "not clinically validated, is not complete, and does not recommend starting, stopping or "
            "changing any medication.",
            icon=":material/science:",
        )

    # Flow: medications -> result headline -> details (explanation, context, source, safety note).
    section_header("Medications to check",
                   "From the synthetic medication record. The check runs again whenever the selection changes.",
                   step=1)
    names_to_check = _render_medication_review(patient, patient_id)
    result = check_interactions(names_to_check)
    _render_medication_tiles(patient, patient_id, result)

    section_header("Result", "What the check found in the sample rules.", step=2)
    _render_headline(result)
    _render_summary(patient, result)

    section_header("Interaction details",
                   "Explanation, recorded patient context and source for each match. Data-quality notes come first.",
                   step=3)
    _render_results(result, patient_id)

    _render_result_type_legend()
    _render_dataset_info()


def _render_result_type_legend():
    """Explain the difference between documented rules and ML predictions."""
    with st.expander("How are these results produced?"):
        rule_col, ml_col = st.columns(2)
        rule_col.markdown(
            f"**{RULE_BADGE}** – shown on this page.  \n"
            "A fixed rule copied from official product labelling, with its source. "
            "The same input always gives the same answer."
        )
        ml_col.markdown(
            "**Machine-learning prediction** – *not used on this page.*  \n"
            "A statistical estimate learned from labelled data. An ML model is used only on the "
            "ICU Risk Assessment page; it is not used by this module. The Deterioration page shows "
            "a separately labelled *simulated* score."
        )


# ---------------------------------------------------------------------------
# 1. Current medications
# ---------------------------------------------------------------------------

def _render_medication_review(patient, patient_id):
    """Selection widgets; return the names the user wants checked."""
    patient_names = [name for name in patient_medication_names(patient) if name]
    with card_container("di_inputs"):
        selected = st.multiselect(
            "Medications to include in the check",
            options=patient_names,
            default=patient_names,
            key=f"di_selected_{patient_id}",
        )

        with st.expander("Demo: add extra medication names to test the checker"):
            st.caption(
                "Type hypothetical names separated by commas, e.g. `Ibuprofen, Albuterol, "
                "Unknownium`. This is only for exploring how the checker behaves "
                "(unknown names, duplicates). It is not a prescribing suggestion."
            )
            extra_text = st.text_input("Extra names (optional)", key=f"di_extra_{patient_id}")

    extra = extra_text.split(",") if extra_text else []
    return selected + extra


def _render_medication_tiles(patient, patient_id, result):
    """One tile per recorded medication, with its status in the current check."""
    meds = get_medications(patient_id)
    if meds.empty:
        st.info("No medications are recorded for this synthetic patient.")
        return

    in_check = set(result.checked_drugs)
    involved = {}
    for interaction in result.interactions:
        for drug in (interaction.drug_a, interaction.drug_b):
            involved[drug] = involved.get(drug, 0) + 1

    tiles = []
    for med in meds.itertuples(index=False):
        name = med.Medication
        if not name:
            tiles.append(tile_html("Name not recorded", badges=[("Missing name – not checked", "warning")],
                                   tone="warning"))
            continue
        key = normalize_name(name)
        if not is_known_drug(name):
            badges, tone = [("Not in sample dataset", "warning")], "warning"
        elif key not in in_check:
            badges, tone = [("Excluded from this check", "neutral")], None
        elif key in involved:
            count = involved[key]
            badges, tone = [(f"In {count} documented interaction{'s' if count != 1 else ''}", "warning")], "warning"
        else:
            badges, tone = [("No rule matched", "neutral")], None
        tiles.append(tile_html(
            name,
            subtitle=" · ".join(str(v) for v in (med.Dose, med.Route, med.Frequency) if v),
            detail=med.Indication or "Indication not recorded",
            badges=badges,
            tone=tone,
        ))
    tile_grid(tiles)


# ---------------------------------------------------------------------------
# 2. Result
# ---------------------------------------------------------------------------

def _render_headline(result):
    """One sentence answer first: what was found, before the counts and the details."""
    count = len(result.interactions)
    if not result.has_input:
        tone, headline = "neutral", "No medications selected"
        detail = "Select medications above to check for interactions."
    elif count:
        tone = "warning"
        headline = f"{count} documented interaction{'s' if count != 1 else ''} found"
        pairs = ", ".join(f"{display_name(i.drug_a)} + {display_name(i.drug_b)}" for i in result.interactions)
        detail = f"{pairs}. The sample rules carry no severity grade; each match is explained below."
    else:
        tone, headline = "info", "No documented interactions found"
        n = len(result.checked_drugs)
        detail = (f"{n} medication{'s' if n != 1 else ''} checked against {len(INTERACTION_RULES)} sample rules. "
                  f"This does not mean the combination is safe – the sample covers only "
                  f"{len(INTERACTION_RULES)} documented pairs.")
    with card_container("di_headline", tone=tone):
        summary_block("Interaction check", headline, detail=detail,
                      badges=[(RULE_BADGE, "info", False)], tone="warning" if count else None)


def _render_summary(patient, result):
    count = len(result.interactions)
    not_checked = len(result.unknown_drugs)
    metric_grid([
        metric_card_html("Medications on record", len(patient_medication_names(patient))),
        metric_card_html(
            "Medications checked", len(result.checked_drugs),
            caption=f"against {len(INTERACTION_RULES)} sample rules",
        ),
        metric_card_html(
            "Documented interactions", count,
            status="Found" if count else "None matched", status_tone="warning" if count else "neutral",
        ),
        metric_card_html(
            "Not checked", not_checked + len(result.duplicates),
            status="Review list" if not_checked or result.duplicates else "None",
            status_tone="warning" if not_checked or result.duplicates else "neutral",
            caption=f"{not_checked} unknown · {len(result.duplicates)} duplicate",
        ),
        metric_card_html(
            "Severity", "Not graded",
            caption="The sample rules carry no severity grade by design.",
        ),
    ])


# ---------------------------------------------------------------------------
# 3–5. Results, no-interaction state and edge cases
# ---------------------------------------------------------------------------

def _render_results(result, patient_id):
    if not result.has_input:
        st.info("**No medications selected.** Select medications above to check for interactions.",
                icon=":material/info:")
        return

    _render_checked_list(result)

    # Data-quality notes come first so they are not missed.
    if result.unknown_drugs:
        st.warning(
            "**Not in the sample dataset – could not be checked:** "
            + ", ".join(result.unknown_drugs)
            + ". The checker cannot say anything about these names.",
            icon=":material/help:",
        )
    if result.duplicates:
        st.warning(
            "**Listed more than once:** "
            + ", ".join(display_name(d) for d in result.duplicates)
            + ". Each drug was checked only once. Duplicate entries in a real "
            "medication list would need to be reviewed by a clinician.",
            icon=":material/content_copy:",
        )
    if result.blank_entries:
        st.caption(f"{result.blank_entries} empty entr{'y was' if result.blank_entries == 1 else 'ies were'} skipped.")

    if not result.interactions:
        st.caption("No interaction details to show: no documented rule matched the medications checked.")
        return

    labs = get_lab_results(patient_id)
    latest_vitals = load_vitals(patient_id).iloc[-1]
    for interaction in result.interactions:
        render_interaction_card(interaction, labs, latest_vitals)


def _render_checked_list(result):
    """Every name that went into the check, with what happened to it."""
    chips = [(display_name(d), "neutral", False) for d in result.checked_drugs]
    chips += [(f"{name} · not checked", "warning", False) for name in result.unknown_drugs]
    chips += [(f"{display_name(d)} · duplicate", "warning", False) for d in result.duplicates]
    label("Checked against the sample rules")
    badge_row(chips)


def render_interaction_card(interaction, labs, latest_vitals):
    """One alert card per detected interaction (also used by the Digital Twin page)."""
    pair = f"{display_name(interaction.drug_a)} + {display_name(interaction.drug_b)}"
    with card_container(f"di_{interaction.drug_a}_{interaction.drug_b}", tone="warning", edge="left"):
        badge_row([
            (interaction.concern, "warning"),
            (RULE_BADGE, "info", False),
            (SEVERITY_NOT_GRADED, "neutral", False),
        ])
        st.markdown(f"#### {pair}")
        st.markdown(f"**What this means:** {interaction.explanation}")

        context = patient_context(interaction, labs, latest_vitals, VITAL_SIGNS)
        if context:
            label("Relevant patient context · recorded synthetic values, not interpreted")
            fact_rows([
                (row["Item"], row["Value"], row["Note"], row["Value"] == "Not recorded") for row in context
            ])
        else:
            st.caption("No patient values are listed as context for this rule.")

        label("Evidence source")
        st.markdown("\n".join(
            f"- [{source['title']}]({source['url']}) – {source['section']}" for source in interaction.sources
        ))
        st.caption(
            "Status: documented rule match (matched on drug names only). "
            "Not a recommendation to start, stop or change any medication."
        )


def _render_dataset_info():
    with st.expander(f"About the sample dataset ({len(INTERACTION_RULES)} rules, {len(KNOWN_DRUGS)} drugs)"):
        rows = [
            {
                "Drug pair": " + ".join(display_name(d) for d in rule["drugs"]),
                "Concern": rule["concern"],
                "Source": "; ".join(s["title"] for s in rule["sources"]),
            }
            for rule in INTERACTION_RULES
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Recognised drug names: " + ", ".join(sorted(display_name(d) for d in KNOWN_DRUGS)) + ". "
            "Rules were taken from official product labelling; labelling changes over time, "
            "so check the current version of each source."
        )
