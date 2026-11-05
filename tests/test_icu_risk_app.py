"""
Headless tests for the ICU Risk Assessment page (views/icu_risk.py) using
Streamlit's AppTest. The page must pass inputs to ml/feature_adapter.py and
ml/inference.py unchanged and never present the score as a probability.
"""

import re

from streamlit.testing.v1 import AppTest

from ml import feature_adapter
from views.icu_risk import EXAMPLE, DISCLAIMER

APP_PATH = "../app.py"
PAGE = "ICU Risk Assessment"
REQUIRED = {"icu_unittype": "MICU", "icu_unitstaytype": "admit",
            "icu_intubated": "No", "icu_vent": "No", "icu_dialysis": "No"}


def _run(app):
    app.run()
    assert not app.exception, app.exception
    return app


def _open():
    app = _run(AppTest.from_file(APP_PATH, default_timeout=60))
    app.session_state["page"] = PAGE
    return _run(app)


def _submit(app, values=None):
    for key, value in (values or {}).items():
        app.session_state[key] = value
    _run(app)
    app.button(key="icu_submit").click()
    return _run(app)


def _text(app):
    parts = [m.value for m in app.markdown] + [c.value for c in app.caption]
    parts += [e.value for e in app.error] + [w.value for w in app.warning] + [i.value for i in app.info]
    parts += [m.label for m in app.metric] + [str(m.value) for m in app.metric]
    return " ".join(parts)


def _score(app):
    return next(m.value for m in app.metric if m.label == "ICU risk score")


def _assert_no_probability_wording(app):
    text = _text(app).lower()
    for phrase in ["probability of death", "chance of death", "mortality probability", "% chance"]:
        assert phrase not in text
    assert not any("probab" in m.label.lower() for m in app.metric)
    assert not re.search(r"\d\s*%", _score(app) if any(m.label == "ICU risk score" for m in app.metric) else "")


def test_page_loads_with_disclaimer_and_no_result():
    app = _open()
    assert app.title[0].value == PAGE
    text = _text(app)
    assert DISCLAIMER in text and "For academic demonstration only" in text
    assert "No assessment yet" in text
    assert not any(m.label == "ICU risk score" for m in app.metric)


def test_example_input_produces_score_matching_ml_layer():
    app = _open()
    app.button(key="icu_load_example").click()
    app = _submit(_run(app))
    text = _text(app)
    assert "LOWER" in text or "ELEVATED" in text
    assert "Demonstration model category" in text
    assert DISCLAIMER in text
    _assert_no_probability_wording(app)

    clinical = {f: None for f in feature_adapter.CLINICAL_FIELDS}
    for field, value in EXAMPLE.items():
        if isinstance(value, str) and field not in ("gender", "unittype", "unitstaytype") \
                and field not in feature_adapter.APACHE_FLAGS:
            value = [float(v) for v in value.split(",")]
        if field in feature_adapter.APACHE_FLAGS:
            value = value == "Yes"
        clinical[field] = value
    expected = feature_adapter.predict_from_clinical(clinical)
    shown = _score(app)
    assert shown == ("< 0.01" if expected["risk_score"] < 0.005 else f"{expected['risk_score']:.2f}")
    assert expected["risk_category"].upper() in text


def test_missing_optional_measurements_and_single_reading():
    app = _submit(_open(), {**REQUIRED, "icu_heartrate": "112"})
    text = _text(app)
    assert _score(app)
    assert "1 / 23 recorded" in text
    assert "Limited observation data may affect model output" in text
    assert "single reading was entered for Heart rate" in text
    assert "Not recorded" in text and "Lactate" in text


def test_multiple_readings_are_aggregated_by_the_adapter():
    single = _submit(_open(), {**REQUIRED, "icu_heartrate": "100", "icu_sao2": "95"})
    multi = _submit(_open(), {**REQUIRED, "icu_heartrate": "80, 100, 120", "icu_sao2": "95, 90, 97"})
    assert "a single reading was entered" not in _text(multi)
    expected = feature_adapter.predict_from_clinical(
        {**{f: None for f in feature_adapter.CLINICAL_FIELDS}, "unittype": "MICU", "unitstaytype": "admit",
         "intubated": False, "vent": False, "dialysis": False,
         "heartrate": [80, 100, 120], "sao2": [95, 90, 97]})
    assert _score(multi) == ("< 0.01" if expected["risk_score"] < 0.005 else f"{expected['risk_score']:.2f}")
    assert _score(single) is not None


def test_empty_record_rejected():
    app = _submit(_open())
    errors = " ".join(e.value for e in app.error)
    assert "clinical record is empty" in errors
    assert not any(m.label == "ICU risk score" for m in app.metric)


def test_missing_required_fields_reported_with_form_labels():
    app = _submit(_open(), {"icu_heartrate": "90"})
    errors = " ".join(e.value for e in app.error)
    assert "5 problems found" in errors
    for name in ["Unit type", "ICU stay type", "Intubated", "Ventilated", "Dialysis"]:
        assert f"{name} is required." in errors
    assert "unitstaytype" not in errors


def test_invalid_numbers_and_units_rejected_without_traceback():
    cases = {
        "icu_heartrate": ("fast", "is not a number"),
        "icu_sao2": ("150", "outside the allowed range"),
        "icu_temperature": ("101.5", "Enter temperature in °C"),
        "icu_lactate": ("2.1, x", "is not a number"),
    }
    for key, (value, message) in cases.items():
        app = _submit(_open(), {**REQUIRED, key: value})
        errors = " ".join(e.value for e in app.error)
        assert message in errors, (key, errors)
        assert "Traceback" not in errors
        assert not any(m.label == "ICU risk score" for m in app.metric)


def test_vitals_without_measurements_rejected():
    app = _submit(_open(), {**REQUIRED, "icu_age": 70.0, "icu_gender": "Female"})
    assert "No vital signs or laboratory values" in " ".join(e.value for e in app.error)


def test_clear_form_resets_inputs_and_result():
    app = _open()
    app.button(key="icu_load_example").click()
    app = _submit(_run(app))
    assert any(m.label == "ICU risk score" for m in app.metric)
    app.button(key="icu_clear").click()
    app = _run(app)
    assert app.session_state["icu_heartrate"] == ""
    assert app.session_state["icu_unittype"] is None
    assert "No assessment yet" in _text(app)


def test_all_problems_reported_together_and_model_not_run():
    app = _submit(_open(), {"icu_heartrate": "fast", "icu_sao2": "150", "icu_temperature": "101.5"})
    errors = [e.value for e in app.error]
    assert len(errors) == 1  # one summary block, no duplicates
    text = errors[0]
    assert "8 problems found" in text and "nothing was assessed" in text
    assert "Heart rate: 'fast' is not a number." in text
    assert "SpO₂: 150 % is outside the allowed range (20–100 %)." in text
    assert "Enter temperature in °C" in text
    assert "ICU stay type is required." in text
    assert "No vital signs" not in text
    assert "Not assessed" in _text(app)
    assert not any(m.label == "ICU risk score" for m in app.metric)


def test_sidebar_status_and_selector_caption():
    app = _open()
    sidebar = " ".join(m.value for m in app.sidebar.markdown) + " ".join(c.value for c in app.sidebar.caption)
    assert "ICU risk model" in sidebar and "Trained · Logistic Regression v0.1.0" in sidebar
    assert "Deterioration engine" in sidebar and "Simulated score" in sidebar
    assert "AI Engine" not in sidebar
    assert "Synthetic patients + eICU demo data (ICU model)" in sidebar
    assert "Synthetic patient selector is not used on this page" in sidebar
    app.session_state["page"] = "Drug Interaction Checker"
    app = _run(app)
    assert "Synthetic patient selector is not used" not in " ".join(c.value for c in app.sidebar.caption)


def test_no_page_claims_the_app_has_no_trained_model():
    app = _open()
    for page in ["Drug Interaction Checker", "Deterioration Prediction", "Patient Digital Twin"]:
        app.session_state["page"] = page
        app = _run(app)
        text = " ".join(m.value for m in app.markdown) + " ".join(e.value for e in app.error)
        assert "No ML model is used anywhere" not in text
        assert "none in this project" not in text
        assert "ICU Risk Assessment page" in text, page


def test_completeness_count_before_button_updates_live():
    app = _open()
    assert "0 / 23 measurements entered" in _text(app)
    app.session_state["icu_heartrate"] = "90, 95"
    app.session_state["icu_lactate"] = "2.1"
    app = _run(app)
    assert "2 / 23 measurements entered" in _text(app)
    assert not any(m.label == "ICU risk score" for m in app.metric)  # entering data never runs the model


def test_result_marked_out_of_date_after_inputs_change():
    app = _submit(_open(), {**REQUIRED, "icu_heartrate": "90, 100"})
    before = _score(app)
    assert "inputs have changed" not in _text(app)
    app.session_state["icu_heartrate"] = "140, 150"
    app = _run(app)
    assert "inputs have changed" in _text(app)
    assert _score(app) == before  # the shown result is unchanged until reassessed
    app.button(key="icu_submit").click()
    app = _run(app)
    assert "inputs have changed" not in _text(app)
