"""
AEGISCARE AI – Integrated Healthcare Safety Intelligence (demo application).

Run with:
    streamlit run app.py

This file only sets up the page, the global styling, the sidebar
(navigation + patient selector + system status) and then hands off to the
selected page module in `views/`.
"""

import streamlit as st

from data.drug_interaction_rules import INTERACTION_RULES
from data.synthetic_patients import list_patients
from ui.components import demo_mode_note, label, nav_item, sidebar_brand, system_status
from ui.styles import inject_css
from views import dashboard, deterioration, digital_twin, drug_interactions, icu_risk

st.set_page_config(
    page_title="AEGISCARE AI",
    page_icon=":material/health_and_safety:",
    layout="wide",
)

PAGES = [
    "Patient Dashboard",
    "Drug Interaction Checker",
    "Deterioration Prediction",
    "ICU Risk Assessment",
    "Patient Digital Twin",
]
# Material Symbols shown before each navigation label.
PAGE_ICONS = {
    "Patient Dashboard": ":material/space_dashboard:",
    "Drug Interaction Checker": ":material/medication:",
    "Deterioration Prediction": ":material/monitor_heart:",
    "ICU Risk Assessment": ":material/ecg_heart:",
    "Patient Digital Twin": ":material/hub:",
}
# Sidebar groups: (group label, [(page, short label shown in the sidebar)]).
NAV_GROUPS = [
    ("Overview", [("Patient Dashboard", "Dashboard")]),
    ("Patient safety", [
        ("Drug Interaction Checker", "Drug Interactions"),
        ("Deterioration Prediction", "Deterioration (simulated)"),
        ("ICU Risk Assessment", "ICU Risk (trained model)"),
    ]),
    ("Simulation", [("Patient Digital Twin", "Patient Digital Twin")]),
]


def _go_to_page(page_name):
    """Nav button callback: set the page before the script reruns."""
    st.session_state["page"] = page_name


def _render_navigation():
    """Grouped navigation buttons plus the hidden radio that holds the page."""
    current = st.session_state.get("page", PAGES[0])
    with st.container(key="iss_nav"):
        for group, items in NAV_GROUPS:
            label(group)
            for page_name, short in items:
                nav_item(
                    short, PAGE_ICONS[page_name], active=page_name == current,
                    on_click=_go_to_page, args=(page_name,),
                    key=f"nav_{page_name.lower().replace(' ', '_')}",
                )

    # The radio is still the single source of the selected page (other pages
    # and the tests set st.session_state["page"]); it is hidden by CSS.
    with st.container(key="iss_nav_state"):
        return st.radio(
            "Navigation",
            PAGES,
            format_func=lambda name: f"{PAGE_ICONS[name]}  {name}",
            key="page",
            label_visibility="collapsed",
        )


def _render_system_status(patient_count):
    icu_status, icu_tone = icu_risk.model_status()
    label("System status")
    system_status([
        # The Deterioration module is a simulated score on synthetic patients; only the
        # ICU Risk Assessment page uses a trained model (shown as unavailable if it cannot load).
        ("Deterioration engine", "Simulated score", "neutral"),
        ("ICU risk model", icu_status, icu_tone),
        ("Interaction rules", f"Documented rules · {len(INTERACTION_RULES)} loaded", "info"),
        ("Patient data", f"{patient_count} synthetic patients", "info"),
    ])
    demo_mode_note("Demo Mode", "Synthetic patients + eICU demo data (ICU model). "
                                "Educational demonstration · Not a medical device.")


def sidebar():
    """Draw the sidebar and return (selected page, selected patient id)."""
    with st.sidebar:
        sidebar_brand()
        st.divider()

        page = _render_navigation()

        patients = list_patients()
        labels = {
            row.patient_id: f"{row.display_name.replace('Synthetic ', '')} · {row.age}{row.sex[0]} ({row.patient_id})"
            for row in patients.itertuples()
        }
        st.divider()
        patient_id = st.selectbox(
            "Active patient (synthetic)",
            options=list(labels),
            format_func=labels.get,
            key="patient_id",
        )
        if page == "ICU Risk Assessment":
            st.caption("Synthetic patient selector is not used on this page. Enter ICU measurements below.")

        st.divider()
        _render_system_status(len(patients))
    return page, patient_id


def main():
    inject_css()
    page, patient_id = sidebar()
    if page == "Patient Dashboard":
        dashboard.render(patient_id)
    elif page == "Drug Interaction Checker":
        drug_interactions.render(patient_id)
    elif page == "Deterioration Prediction":
        deterioration.render(patient_id)
    elif page == "ICU Risk Assessment":
        icu_risk.render(patient_id)
    elif page == "Patient Digital Twin":
        digital_twin.render(patient_id)


main()
