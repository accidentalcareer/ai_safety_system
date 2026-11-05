"""Headless tests of the Patient Digital Twin page using Streamlit's AppTest."""

from streamlit.testing.v1 import AppTest

from views.digital_twin import REGISTRY_KEY

APP_PATH = "../app.py"


def _run(app):
    app.run()
    assert not app.exception, app.exception
    return app


def _open_twin(patient_id="SYN-001"):
    app = _run(AppTest.from_file(APP_PATH, default_timeout=30))
    app.session_state["page"] = "Patient Digital Twin"
    app.session_state["patient_id"] = patient_id
    return _run(app)


def _twin(app, patient_id):
    return app.session_state[REGISTRY_KEY].get(patient_id)


def _metric(app, label):
    return next(m.value for m in app.metric if m.label == label)


def _click(app, key):
    app.button(key=key).click()
    return _run(app)


def test_page_loads_for_every_patient_with_clear_labels():
    for patient_id in ["SYN-001", "SYN-002", "SYN-003"]:
        app = _open_twin(patient_id)
        assert app.title[0].value.endswith("Patient Digital Twin")
        assert any(patient_id in caption.value for caption in app.caption)
        assert "not clinical decision support" in app.warning[0].value
        labels = [m.label for m in app.metric]
        assert "Demo deviation score (simulated)" in labels
        assert not any("probab" in label.lower() for label in labels)
        assert len(app.tabs) == 4


def test_add_medication_updates_state_and_logs_once_across_reruns():
    app = _open_twin("SYN-001")
    app.text_input(key="twin_SYN-001_med_typed").input("Ibuprofen")
    _click(app, "twin_SYN-001_btn_add")

    twin = _twin(app, "SYN-001")
    assert "Ibuprofen" in twin.medication_names
    assert [e.change_type for e in twin.events] == ["Twin created", "Medication added"]
    assert _metric(app, "Documented rule matches") == "4"
    assert any("Recorded event #2" in s.value for s in app.success)
    assert any("needs reassessment" in w.value for w in app.warning)

    # Normal reruns (e.g. any other widget interaction) must not log again.
    for _ in range(3):
        _run(app)
    assert len(twin.events) == 2
    assert _metric(app, "Timeline events") == "2"


def test_remove_medication_from_selectbox():
    app = _open_twin("SYN-001")
    app.selectbox(key="twin_SYN-001_med_remove").set_value("Aspirin")
    _click(app, "twin_SYN-001_btn_remove")
    twin = _twin(app, "SYN-001")
    assert "Aspirin" not in twin.medication_names
    assert _metric(app, "Documented rule matches") == "0"


def test_vital_change_updates_score_and_timeline():
    app = _open_twin("SYN-003")
    before = float(_metric(app, "Demo deviation score (simulated)"))
    app.selectbox(key="twin_SYN-003_vital_key").set_value("resp_rate")
    _run(app)
    app.number_input(key="twin_SYN-003_vital_value_resp_rate").set_value(32.0)
    _click(app, "twin_SYN-003_btn_vital")

    assert float(_metric(app, "Demo deviation score (simulated)")) > before
    event = _twin(app, "SYN-003").events[-1]
    assert event.change_type == "Vital sign changed" and event.new_value == "32 /min"
    timeline = next(df.value for df in app.dataframe if "Modules reassessed" in df.value.columns)
    assert timeline.iloc[0]["Change"] == "Vital sign changed"


def test_invalid_vital_shows_error_and_records_nothing():
    app = _open_twin("SYN-002")
    app.selectbox(key="twin_SYN-002_vital_key").set_value("spo2")
    _run(app)
    app.number_input(key="twin_SYN-002_vital_value_spo2").set_value(150.0)
    _click(app, "twin_SYN-002_btn_vital")
    assert any("outside the data-entry limits" in e.value for e in app.error)
    assert len(_twin(app, "SYN-002").events) == 1


def test_lab_change_and_manual_reassessment():
    app = _open_twin("SYN-001")
    app.text_input(key="twin_SYN-001_med_typed").input("Ibuprofen")
    _click(app, "twin_SYN-001_btn_add")
    app.selectbox(key="twin_SYN-001_lab_test").set_value("Potassium")
    _run(app)
    app.number_input(key="twin_SYN-001_lab_value_Potassium").set_value(6.0)
    _click(app, "twin_SYN-001_btn_lab")
    _click(app, "twin_SYN-001_btn_reassess")

    changes = [e.change_type for e in _twin(app, "SYN-001").events]
    assert changes == ["Twin created", "Medication added", "Lab value changed", "Manual reassessment"]
    assert not any("needs reassessment" in w.value for w in app.warning)


def test_patients_keep_separate_state_when_switching():
    app = _open_twin("SYN-001")
    app.text_input(key="twin_SYN-001_med_typed").input("Ibuprofen")
    _click(app, "twin_SYN-001_btn_add")

    app.session_state["patient_id"] = "SYN-002"
    _run(app)
    assert "Ibuprofen" not in _twin(app, "SYN-002").medication_names
    assert _metric(app, "Timeline events") == "1"

    app.session_state["patient_id"] = "SYN-001"
    _run(app)
    assert "Ibuprofen" in _twin(app, "SYN-001").medication_names
    assert _metric(app, "Timeline events") == "2"


def test_reset_restores_original_state():
    app = _open_twin("SYN-001")
    original_score = _metric(app, "Demo deviation score (simulated)")
    app.selectbox(key="twin_SYN-001_vital_key").set_value("heart_rate")
    _run(app)
    app.number_input(key="twin_SYN-001_vital_value_heart_rate").set_value(140.0)
    _click(app, "twin_SYN-001_btn_vital")
    assert _metric(app, "Demo deviation score (simulated)") != original_score

    _click(app, "twin_SYN-001_btn_reset")
    assert _metric(app, "Demo deviation score (simulated)") == original_score
    assert _twin(app, "SYN-001").events[-1].change_type == "Reset"


def test_other_pages_still_show_original_record_after_twin_edits():
    app = _open_twin("SYN-001")
    app.selectbox(key="twin_SYN-001_med_remove").set_value("Aspirin")
    _click(app, "twin_SYN-001_btn_remove")
    app.session_state["page"] = "Drug Interaction Checker"
    _run(app)
    assert "Warfarin + Aspirin" in " ".join(md.value for md in app.markdown)
