# Integrated AI Safety Systems in Healthcare – Demo

A local Streamlit demonstration built for a college project.

> ⚠️ **All data in this app is synthetic.** The three patients, their medications,
> vital signs and lab results are invented. The app is an educational demo, is not a
> medical device, and must not be used for clinical decisions.

## Milestone status

| Module | Status |
|---|---|
| 🩺 Patient Dashboard | ✅ Implemented |
| 💊 Drug Interaction Checker | ✅ Implemented (milestone 2) |
| 📉 Deterioration Prediction | ✅ Implemented in **simulated demonstration mode** (milestone 3) |
| 🧬 Patient Digital Twin | ✅ Implemented – integrates the modules above on a simulated patient copy |

The **Patient Dashboard** includes:

- A sidebar patient selector with 3 synthetic patients (`SYN-001`, `SYN-002`, `SYN-003`)
- Patient details: age, sex, ward, admission date, diagnoses, allergies
- Latest vital signs (BP, heart rate, SpO₂, respiratory rate, temperature) with change since the previous reading
- Interactive Plotly trend charts with a 24 / 48 / 72-hour window and a choice of vitals
- Current medication list
- Laboratory results with a plain Low / Normal / High range check

The "typical range" bands and Low/High labels are **simple comparisons against
illustrative adult reference ranges**. They are not predictions or clinical interpretations.

## Drug Interaction Checker

- Checks the selected patient's medication list against a **small teaching sample of 9
  documented interaction rules** (`data/drug_interaction_rules.py`). Each rule cites the
  official product labelling it comes from (US FDA labels on DailyMed, an EU SmPC, and an
  FDA science paper).
- Each match shows the drug pair, the type of concern, a plain-language explanation, the
  evidence source, and relevant recorded patient values (e.g. INR, potassium). These values
  are shown for context only and are not interpreted.
- Every result is labelled **📘 Documented rule**. This module contains no machine-learning
  predictions.
- Unknown drug names, duplicates (including aliases such as Albuterol = Salbutamol), blank
  entries and empty lists are reported instead of causing errors.
- **No rule found does not mean safe.** The sample covers only a few pairs, is not clinically
  validated and does not recommend any medication change.

## Deterioration Prediction (simulated demonstration mode)

**Why there is no trained model.** A supervised scikit-learn model needs *labelled* data:
observations plus a recorded outcome. The synthetic records have no outcome labels, so nothing
could be learned or evaluated. Making up labels from a rule and training on them would only teach
the model to copy that rule, so no model is trained. `labeled_data_check()` in
`utils/deterioration.py` makes this decision in code and the page explains it.

**What the page shows**

- Vital-sign trend charts (BP, heart rate, SpO₂, respiratory rate, temperature) with a 24/48/72 h window
- Lab results plotted relative to their own reference ranges
- A **🧪 Simulated demonstration score** with a chart and table of contributing factors
- Missing-data and invalid-input warnings, the number of inputs available, and the score over time
- A "what-if" panel to edit the latest readings or remove readings (for demos only)
- A limitations section

**How the demo score is calculated** (deliberately simple and transparent):

1. *Outside range*: how far the latest value is outside its range, in multiples of the range width
   (inside the range = 0). Vitals use the dashboard's illustrative ranges; labs use their own
   reference range from the record.
2. *Worsening*: how much that distance grew over the last 24 h (vital signs only).
3. Each part is capped at 1. Points = outside range + worsening. Score = sum of all points.

The ranges, equal weights and caps are teaching choices, not validated thresholds. The score
**is not a probability or a clinical prediction**, and no Low/High risk band is shown because
no validated cut-off exists. Missing inputs add no points, so incomplete data lowers the score.

**What a real trained model would need** (future work, if labelled data were added):
- *Target*: a binary label per observation time, e.g. "an agreed deterioration event (such as ICU
  transfer) occurred within the next 24 hours", stored as `deterioration_outcome`.
- *Model*: a baseline such as scikit-learn `LogisticRegression` on the same features
  (latest values and 24 h changes), with imputation and scaling in a `Pipeline`.
- *Evaluation*: split by **patient** (not by row) so that no patient appears in both
  training and test data, then report AUROC, precision/recall and a calibration plot
  on the held-out patients.

## Patient Digital Twin

An editable, **simulated** copy of the selected synthetic patient that re-runs the existing
modules after each change. Logic: `utils/digital_twin.py` (no Streamlit code). Page:
`views/digital_twin.py`.

| Change | Drug Interaction Checker (📘 documented rule) | Deterioration score (🧪 simulated) |
|---|---|---|
| Add / remove a medication | Re-run | **Not recomputed** – marked *Needs reassessment*, because the score has no medication inputs (`USES_MEDICATION_DATA = False`) |
| Change a vital sign (latest reading) | Re-run **only** if a matched rule lists that vital as context (e.g. Sertraline + Sumatriptan → heart rate) | Recomputed |
| Change a lab value | Re-run **only** if a matched rule lists that lab as context (e.g. Warfarin + Aspirin → INR) | Recomputed only for the `DEMO_LABS` it uses |

- Interaction rules match on drug **names** only; context values are refreshed for display but never
  change which rules match.
- Every change adds one timeline event: time, patient ID, change type, previous → new value, modules
  reassessed and why, resulting status, and output source. No output is labelled as an evaluated
  model because the project has none.
- Each patient has its own twin in Streamlit session state. Changes run in button callbacks, so a
  normal rerun never logs an event twice. **Reset** restores the original sample values and keeps
  the timeline.
- Invalid input (blank or duplicate medication names, non-numeric or implausible values, unknown
  tests) is rejected with a message and leaves the state unchanged. Unknown medication names are
  added but reported as *not checked*.
- The other pages always show the original synthetic record.

## Project structure

```
healthcare_safety_app/
├── app.py                      # Entry point: page setup, sidebar navigation, patient selector
├── data/
│   ├── synthetic_patients.py   # Synthetic patients, vitals generator, labs, reference ranges
│   └── drug_interaction_rules.py # Sample interaction rules with evidence sources
├── utils/
│   ├── charts.py               # Plotly chart helpers
│   ├── drug_interactions.py    # Interaction checking logic (no Streamlit code)
│   ├── deterioration.py        # Demo score: validation, preprocessing, scoring (no Streamlit code)
│   └── digital_twin.py         # Digital twin state, change workflows, timeline (no Streamlit code)
├── views/
│   ├── dashboard.py            # Patient Dashboard page
│   ├── drug_interactions.py    # Drug Interaction Checker page
│   ├── deterioration.py        # Deterioration Prediction page (simulated mode)
│   └── digital_twin.py         # Patient Digital Twin page
├── tests/
│   ├── test_synthetic_data.py  # Unit tests for the data module
│   ├── test_drug_interactions.py # Unit tests for the interaction checker
│   ├── test_deterioration.py   # Unit tests for the demo score
│   ├── test_digital_twin.py    # Unit tests for the twin workflows
│   ├── test_digital_twin_app.py # Headless tests of the Digital Twin page
│   └── test_app.py             # Headless smoke tests of the Streamlit app
├── requirements.txt
└── pytest.ini
```

## Installation (Windows)

Requires Python 3.10 or newer. Open PowerShell in this folder, then run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

macOS / Linux: use `source .venv/bin/activate` instead of the second line.

## Run the app

```powershell
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Press `Ctrl+C` in the terminal to stop the app.

## Run the tests

```powershell
python -m pytest
```

## How the synthetic data is made

Each patient has a start value, end value, and noise level for every vital sign
(see `PATIENTS` in `data/synthetic_patients.py`). The generator draws a straight line
from start to end over 72 hours (one reading every 2 hours) and adds random noise
from a fixed seed, so the numbers are identical on every run. To change a patient,
edit their profile in that file.
