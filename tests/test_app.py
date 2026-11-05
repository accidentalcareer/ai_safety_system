"""
Smoke tests that run the real Streamlit app headlessly using
Streamlit's built-in AppTest helper.
"""

from streamlit.testing.v1 import AppTest

APP_PATH = "../app.py"


def _run(app=None):
    app = app or AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    assert not app.exception, app.exception
    return app


def test_dashboard_loads_for_every_patient():
    app = _run()
    selector = app.sidebar.selectbox(key="patient_id")
    assert len(selector.options) == 3
    for patient_id in ["SYN-001", "SYN-002", "SYN-003"]:
        app.session_state["patient_id"] = patient_id
        app.run()
        assert not app.exception, app.exception
        assert any(patient_id in caption.value for caption in app.caption)
        # 5 latest-vital cards are shown
        assert len(app.metric) == 5


def test_navigation_to_every_page():
    app = _run()
    nav = app.sidebar.radio(key="page")
    expected_titles = ["Synthetic Patient A", "Drug Interaction Analysis", "Patient Deterioration Intelligence", "ICU Risk Assessment", "Patient Digital Twin"]
    assert len(nav.options) == len(expected_titles)
    for index, expected in enumerate(expected_titles):
        app.session_state["page"] = nav.options[index].split("  ", 1)[1]
        app.run()
        assert not app.exception, app.exception
        assert app.title[0].value.endswith(expected)


def _open_checker(patient_id):
    app = _run()  # first run creates the widgets, then we switch page
    app.session_state["page"] = "Drug Interaction Checker"
    app.session_state["patient_id"] = patient_id
    return _run(app)


def _all_markdown(app):
    return " ".join(md.value for md in app.markdown)


def test_interaction_checker_shows_documented_rules():
    app = _open_checker("SYN-001")
    text = _all_markdown(app)
    assert "Warfarin + Aspirin" in text
    assert "Documented rule" in text
    assert "Evidence source" in text


def test_interaction_checker_handles_extra_unknown_and_duplicate_names():
    app = _open_checker("SYN-002")
    app.text_input(key="di_extra_SYN-002").input("Unknownium, Salbutamol, ").run()
    assert not app.exception, app.exception
    warnings = " ".join(w.value for w in app.warning)
    assert "Unknownium" in warnings
    assert "Listed more than once" in warnings


def test_interaction_checker_with_no_medications_selected():
    app = _open_checker("SYN-003")
    app.multiselect(key="di_selected_SYN-003").set_value([]).run()
    assert not app.exception, app.exception
    assert any("No medications selected" in info.value for info in app.info)


def test_dashboard_button_opens_interaction_checker():
    app = _run()
    app.button(key="open_interaction_checker").click().run()
    assert not app.exception, app.exception
    assert app.title[0].value.endswith("Drug Interaction Analysis")


# --- Deterioration Prediction page -----------------------------------------

def _open_deterioration(patient_id):
    app = _run()
    app.session_state["page"] = "Deterioration Prediction"
    app.session_state["patient_id"] = patient_id
    return _run(app)


def test_deterioration_page_loads_for_every_patient():
    for patient_id in ["SYN-001", "SYN-002", "SYN-003"]:
        app = _open_deterioration(patient_id)
        assert any(patient_id in caption.value for caption in app.caption)
        # Clearly labelled as simulated, never as a trained model / probability.
        banner = app.error[0].value
        assert "Simulated demonstration mode" in banner and "not a trained model" in banner
        labels = [m.label for m in app.metric]
        assert "Demo deviation score" in labels
        assert not any("probab" in label.lower() for label in labels)
        assert any(h.value.startswith("Limitations") for h in app.subheader)


def test_deterioration_page_reports_missing_lab():
    app = _open_deterioration("SYN-001")  # no C-reactive protein recorded
    warnings = " ".join(w.value for w in app.warning)
    assert "Missing data" in warnings and "C-reactive protein" in warnings


def test_deterioration_what_if_edit_changes_score():
    app = _open_deterioration("SYN-003")
    before = next(m.value for m in app.metric if m.label == "Demo deviation score")
    app.number_input(key="det_edit_SYN-003_resp_rate").set_value(40.0).run()
    assert not app.exception, app.exception
    after = next(m.value for m in app.metric if m.label == "Demo deviation score")
    assert float(after) > float(before)
    assert any("edited" in info.value for info in app.info)


def test_deterioration_invalid_and_missing_inputs_are_reported():
    app = _open_deterioration("SYN-002")
    app.number_input(key="det_edit_SYN-002_spo2").set_value(150.0).run()
    app.multiselect(key="det_remove_SYN-002").set_value(["heart_rate"]).run()
    assert not app.exception, app.exception
    warnings = " ".join(w.value for w in app.warning)
    assert "Invalid input ignored" in warnings
    assert "Heart rate: no valid readings" in warnings
