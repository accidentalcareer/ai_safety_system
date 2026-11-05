# AEGISCARE – ICU mortality ML pipeline (proof of concept)

This directory is self-contained and does not touch the Streamlit app, the Drug
Interaction Checker, the Digital Twin, or their tests.

```
ml/
├── data/
│   ├── raw/                      # eICU demo CSVs – read-only, never modified
│   └── processed/
│       └── mortality_training.csv
├── models/                       # saved pipeline + model_metadata.json
├── results/                      # metrics.csv, split.csv, plots
├── data_audit.py                 # read-only audit of the raw tables
├── preprocessing.py              # builds the patient-level dataset
├── train.py                      # trains/compares the 3 models, saves the selected one
├── evaluate.py                   # metrics/plots; re-checks the saved model
├── inference.py                  # predict_mortality(): validated scoring of one patient
├── test_inference.py             # tests for inference.py
├── feature_adapter.py            # build_model_features(): clinical fields -> 126 features
├── test_feature_adapter.py       # tests for feature_adapter.py
└── README.md
```

```
pip install -r ml/requirements.txt
python ml/data_audit.py       # prints the audit (writes nothing)
python ml/preprocessing.py    # writes ml/data/processed/mortality_training.csv
python ml/train.py            # trains/compares 3 models, writes ml/models/ and ml/results/
python ml/evaluate.py         # reloads the saved model and re-checks its test metrics
python -m pytest ml/test_inference.py   # inference-layer tests
python -m pytest ml/test_feature_adapter.py   # feature-adapter tests
```

Use the project virtual environment (`.venv\Scripts\python.exe` on Windows): it has the exact
library versions the model was saved with (see `library_versions` in `model_metadata.json`).

> **Disclaimer:** this is a college-project proof of concept trained on the eICU *Demo*
> dataset. It is **not** clinically validated and must not be used for patient-care
> decisions. Its output is a demonstration risk estimate, not a diagnosis.

## Source files

eICU Collaborative Research Database **Demo** (`.csv` or `.csv.gz` both accepted):

| File | Used for |
|---|---|
| `apachePatientResult.csv` | target (`actualicumortality`), defines the cohort |
| `patient.csv` | demographics, unit type, stay type, patient id for grouping |
| `apacheApsVar.csv` | `intubated`, `vent`, `dialysis` flags |
| `vitalPeriodic.csv` | HR, SpO2, respiration, temperature, invasive BP |
| `vitalAperiodic.csv` | non-invasive BP |
| `lab.csv` | 13 laboratory values |

**Join key:** `patientunitstayid` (one ICU stay). All joins are left joins onto the cohort.

## Target

`actualicumortality` in `apachePatientResult.csv`:

| Raw value | Rows | Stays | Encoded `icu_mortality` |
|---|---|---|---|
| `ALIVE` | 3,494 | 1,747 | 0 |
| `EXPIRED` | 182 | 91 | 1 |
| missing | 0 | 0 | – |

Each stay has **two rows**, one for APACHE IV and one for IVa. For every stay both
rows give the same outcome (the pipeline asserts this), so one row per stay is kept.
`actualicumortality` also matches `patient.unitdischargestatus` exactly (1:1).

**Cohort:** the 1,838 stays that have an APACHE result. The other 682 stays in
`patient.csv` have no target and are excluded.

## Observation window

Using the whole stay would leak the outcome: whole-stay statistics include vitals recorded
right up to death, and they also encode how long the stay lasted. Features therefore use a
fixed window measured from unit admission:

| Source | Window (minutes from unit admit) |
|---|---|
| Vitals | 0 to 1440 (first 24h) |
| Labs | −360 to 1440 (6h before admission, to capture ED / pre-ICU bloodwork, through 24h) |

All stays are kept, including those that end within 24h. Of the 91 deaths, 27 happen
inside the 24h window, so for those patients the features include data from close to
death. This known limitation is described under [Known issues](#known-issues--warnings).

## Features (126)

### Static (24)
- `age`: eICU stores ages above 89 as `"> 89"`, which is mapped to 90.
- `admissionweight` (kg) and `admissionheight` (cm).
- `age_missing`, `admissionweight_missing`, `admissionheight_missing`.
- One-hot encoding with fixed category lists, so the schema stays stable at inference:
  - `gender_female`, `gender_male`, `gender_missing`
  - `unittype_*` (8): cardiac_icu, ccu_cticu, csicu, cticu, med_surg_icu, micu, neuro_icu, sicu
  - `unitstaytype_*` (4): admit, readmit, stepdown_other, transfer

  An unknown category is encoded as all zeros. A category not seen in the audit raises an error.
- `apache_intubated`, `apache_vent`, `apache_dialysis`: 0/1 flags. The audit found **no
  `-1` sentinels** in these three columns, and the pipeline asserts the values are binary.
  These flags are APACHE first-day values, which fits the 24h window. The other APACHE
  physiology variables are deliberately left out: 25–82% of their values are the `-1`
  "not recorded" sentinel.

### Vitals (50): `mean`, `min`, `max`, `std` plus a `<var>_missing` flag, over the first 24h
`heartrate`, `sao2`, `respiration`, `temperature`, `systemicsystolic`,
`systemicdiastolic`, `systemicmean` (invasive / arterial line), `noninvasivesystolic`,
`noninvasivediastolic`, `noninvasivemean`.

### Labs (52): `first`, `min`, `max` plus a `<lab>_missing` flag

| Feature | Exact eICU `labname` | Not used (and why) |
|---|---|---|
| glucose | `glucose` | `bedside glucose` (fingerstick, a different assay), `glucose - CSF` |
| sodium | `sodium` | `urinary sodium` |
| potassium | `potassium` | |
| chloride | `chloride` | |
| bicarbonate | `bicarbonate` | `HCO3` (ABG-derived), `Total CO2` |
| creatinine | `creatinine` | `urinary creatinine` |
| bun | `BUN` | |
| wbc | `WBC x 1000` | `WBC's in <body fluid>` |
| hemoglobin | `Hgb` | carboxy-/met-/oxy-hemoglobin |
| platelets | `platelets x 1000` | |
| calcium | `calcium` (total) | `ionized calcium` |
| magnesium | `magnesium` | |
| lactate | `lactate` | |

**Why first/min/max only:** within the window the median number of results per stay
is **2** for 11 of the labs and **1** for lactate and magnesium. With so few values, `last`
and `mean` add almost no information beyond `first`, `min` and `max`. `first` captures the
admission state, and `min`/`max` capture the extremes in either direction.

Lab handling details:
- A small number of results are censored strings with no numeric `labresult`
  (`"<0.50"`, `">40"`). There are 55 in the window, and each is parsed to its numeric bound.
- When the same lab appears at the same offset more than once, the row with the latest
  `labresultrevisedoffset` is kept.

## Cleaning (data validity only)

The following ranges are used only to remove physiologically impossible readings, such as
monitor artifacts and unit errors. Values outside them are set to **NaN**, not clipped.
These ranges are never used to create labels.

| Variable | Valid range | Variable | Valid range |
|---|---|---|---|
| age | 0–120 | weight | 20–350 kg |
| height | 100–250 cm | heart rate | 20–300 |
| SpO2 | 20–100 | respiration | 1–100 |
| temperature | 25–45 °C | systolic BP | 20–300 |
| diastolic BP | 5–250 | mean BP | 10–250 |

Labs also have broad validity ranges (see `PLAUSIBLE_RANGES` in `preprocessing.py`).
No lab value in the window fell outside them.

Temperatures between 80 and 115 in the °C column are treated as °F and converted to °C.
1,006 readings were converted.

Values removed in the last run: respiration 3,207 (mostly `0`), SpO2 684, systemic mean 228,
systemic systolic 181, heart rate 128, systemic diastolic 62, temperature 43, NIBP 26–32 each,
height 11, weight 1. Fully duplicated time-series rows are dropped before aggregation.

## Missing values

Rows are **not** dropped for missingness. The strategy:

1. **Preserve missingness.** Each source variable has a `<var>_missing` indicator (1 = no valid
   observation in the window). This is informative here: for example, a missing arterial-line BP
   (`systemicsystolic_missing`) occurs in 83% of survivors and 67% of deaths.
2. **Leave NaN in the CSV.** Numeric NaNs are kept in `mortality_training.csv` on purpose.
   Imputing before the train/test split would leak test-set statistics into training.
3. **Median imputation at training time.** `preprocessing.build_imputer()` returns a
   `SimpleImputer(strategy="median")`. It must be fit on the training split only, ideally
   inside an sklearn `Pipeline`. A `*_std` value is NaN when only one observation exists,
   and the imputer handles that too.

Features still containing NaN: 82 of 126. The highest rates are temperature (93%),
invasive BP (about 83%), lactate (68%), magnesium (44%), bicarbonate (12%), and the remaining
labs (5–9%). Vitals other than temperature and invasive BP are under 10% missing. All
indicator, one-hot and APACHE flag columns are complete.

## Leakage exclusions

None of these appear in the features, and `validate()` asserts it by exact name and by pattern
(`predicted|actual|unabridged|discharge|apachescore|acutephysiology|mortality`):

- Specified: `actualicumortality` (only used to derive the target), `predictedicumortality`,
  `predictedhospitalmortality`, `predictediculos`, `predictedhospitallos`, `predventdays`,
  `actualiculos`, `actualhospitallos`, `actualventdays`, `unabridgedactualventdays`.
- **Additionally flagged and excluded:**
  - `patient.unitdischargestatus` matches the target exactly (1:1).
  - `hospitaldischargestatus`, `unitdischargeoffset` (ICU length of stay), `hospitaldischargeoffset`,
    the discharge locations and times, `hospitaldischargeyear`, and `dischargeweight`
    (51% missing and recorded at discharge).
  - `actualhospitalmortality`, `unabridgedunitlos` and `unabridgedhosplos`.
  - `apachescore` and `acutephysiologyscore`. These are derived severity scores, and APACHE's
    mortality prediction is built from them. They could serve as a benchmark but are not inputs.

## Output: `ml/data/processed/mortality_training.csv`

| | |
|---|---|
| Rows | 1,838 (one per ICU stay; `patientunitstayid` is unique) |
| Columns | 129 = 2 id columns + 1 target + 126 features |
| Id columns (not features) | `patientunitstayid`, `uniquepid` |
| Target | `icu_mortality` |
| Positives (EXPIRED) | 91 (4.95%) |
| Negatives (ALIVE) | 1,747 (95.05%) |
| Distinct patients | 1,512 |

`preprocessing.feature_columns(df)` returns the feature list.

`validate()` checks the following on every run: unique stay ids, target present and in {0, 1},
target and leakage columns absent from the features, all features numeric, no ±inf values, and
no duplicate rows.

## Known issues / warnings

- **Very few positives (91).** Use stratified cross-validation and report PR-AUC alongside ROC-AUC.
  Expect wide confidence intervals.
- **Repeated patients.** 1,838 stays come from 1,512 patients. Train/test splits must be
  **grouped by `uniquepid`**, otherwise the same patient can appear on both sides.
- **Early deaths.** 27 of the 91 deaths occur within the 24h window, so their features include data
  from close to death. A stricter "predict at hour 24" design would keep only stays of at least 24h,
  leaving 1,325 stays and 64 deaths.
- **Temperature** is 93% missing in the vitalPeriodic feed during the first 24h, so its statistics
  are weak. They are kept, with the indicator column.
- **Lab availability time.** Labs are windowed on `labresultoffset` (when the specimen was resulted).
  The final revised value may become available somewhat later (median 39 min).
- **Cohort selection.** The APACHE-scored cohort excludes some stays, such as those without an APACHE
  result, so results generalise to APACHE-eligible stays only.

## Model training (`train.py`, `evaluate.py`)

**Split.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. The first fold is
held out as the test set: 1,470 train / 368 test stays (80/20), 1,208 / 304 patients, 73 / 18
deaths (4.97% / 4.89%).

Groups are **`uniquepid` (patient)**, not `patientunitstayid`. Every stay already has a unique
`patientunitstayid`, so grouping by it would be a plain row-level split, and 416 patients have
more than one stay. `train.py` asserts that no patient and no stay appears in both sets. Both id
columns are excluded from X.

**Models.** Each model is a single sklearn `Pipeline`, fit on training data only:

| Model | Pipeline | Imbalance handling |
|---|---|---|
| Logistic Regression | median impute → drop constant cols → StandardScaler → LR (L2, C=0.1) | `class_weight="balanced"` |
| Random Forest | median impute → drop constant cols → RF (500 trees, min_samples_leaf=3) | `class_weight="balanced_subsample"` |
| XGBoost | drop constant cols → XGB (300 trees, depth 3, lr 0.05); NaN handled natively | `scale_pos_weight` = train neg/pos (19.14) |

Categoricals are already one-hot encoded with fixed categories by `preprocessing.py`, so there is
no learned encoder. The variance filter removes `unitstaytype_stepdown_other`, which is constant
in the training data.

**Model selection** uses only the training set, through grouped 5-fold cross-validation. The
selected model has the lowest mean rank across CV PR-AUC, ROC-AUC, recall, precision and F1, with
ties broken by CV PR-AUC. Accuracy is not a criterion. The test set is used once, for the final
report.

**Decision threshold.** The threshold is the one that maximises F1 on the out-of-fold training
predictions, and it is stored in the metadata. Metrics at 0.5 are also written to `metrics.csv`
for reference.

**Why not accuracy.** About 95% of stays survive, so a model that always predicts "survives"
scores about 95% accuracy while finding zero deaths. On this test set, for example, Random Forest
at threshold 0.5 reaches 95.1% accuracy but recalls only 1 of 18 deaths. PR-AUC, recall,
precision and F1 measure performance on the rare class that matters.

**Outputs**

| Path | Contents |
|---|---|
| `ml/models/aegiscare_mortality_model.pkl` | selected fitted pipeline (joblib); `predict_proba(X[feature_names])` |
| `ml/models/model_metadata.json` | features, threshold, split sizes, seed, metrics, preprocessing, versions |
| `ml/results/metrics.csv` | CV and test metrics for all 3 models |
| `ml/results/split.csv` | train/test assignment per stay |
| `ml/results/roc_curve.png`, `pr_curve.png`, `confusion_matrix_best.png` | evaluation plots |

**Limitations.** The test set has only 18 deaths, so the bootstrap 95% CIs are wide (for example,
PR-AUC roughly 0.14 to 0.70 across the models) and differences between models are not
statistically meaningful. The model was trained on a single demo dataset with no external
validation and no probability calibration.

## Inference (`inference.py`)

```python
from ml.inference import predict_mortality, MortalityInputError   # from the repo root
result = predict_mortality(patient_features)   # dict, pandas Series or 1-row DataFrame
```

**How it works.** The saved pipeline (`aegiscare_mortality_model.pkl`) is loaded once and reused
as-is: median imputation, the constant-feature filter and scaling all happen inside it, exactly as
in training. `inference.py` re-implements no preprocessing. It checks the record and then calls
`predict_proba`. Its scores are identical to calling the pipeline directly (tested on the whole
test split).

**Expected input:** one ICU stay as the **126 features** produced by `preprocessing.py`, using the
names from `inference.feature_names()` (the same list as `model_metadata.json`). Raw vitals and
labs have to be aggregated over the same windows first (first 24h for vitals, −6h to +24h for
labs). Validation raises `MortalityInputError`, whose `errors` attribute lists every problem:

- All 126 keys must be present. A missing key is an error and is never filled in silently.
  Unknown keys are also rejected, which catches typos such as `heart_rate`.
- Indicator, one-hot and APACHE-flag columns must be 0 or 1. Exactly one column must be set in
  each of the gender, unit-type and stay-type groups.
- A measured value may be `None` or NaN **only** when it was not recorded, and then its
  `<var>_missing` flag must be 1. The pipeline imputes such values with the training median,
  just as in training, and the result lists them in `features_not_recorded`.
- Recorded values must be finite numbers inside the cleaning plausibility ranges, with
  min ≤ first/mean ≤ max and std ≥ 0. Temperature must be in °C.
- **Empty or near-empty records are rejected.** An empty dict, `None`, or a record with no vital
  sign and no lab value at all would produce a score built purely from imputed medians. This
  rejects 6 of the 1,838 training stays.

`ModelLoadError` is raised if the model or metadata file is missing, unreadable, or inconsistent.

**Output** (a JSON-serialisable dict):

| Key | Meaning |
|---|---|
| `risk_score` | model output in [0, 1]: an **uncalibrated risk score**, not a probability |
| `risk_category` | `"lower"` or `"elevated"` (demonstration category, see below) |
| `category_description`, `above_threshold`, `decision_threshold` | how the category was assigned |
| `model_name`, `model_version` | e.g. `Logistic Regression`, `0.1.0` |
| `score_type`, `disclaimer` | text to show next to the score |
| `features_not_recorded`, `n_features_not_recorded` | inputs given as blank and imputed by the pipeline |

**Demonstration categories.** The only cut-point is the model's own decision threshold
(`decision_threshold` = 0.810), which was chosen to maximise F1 on out-of-fold training
predictions:

| `risk_score` | `risk_category` |
|---|---|
| < 0.810 | `lower`: below the model decision threshold |
| ≥ 0.810 | `elevated`: at or above the model decision threshold (flagged by the model) |

These are model categories for the demo. They are **not** clinical risk classes, and no clinical
thresholds were defined. On the test set, scores at or above the threshold caught 7 of 18 deaths,
with 15 false alarms.

**Why "risk score" and not "probability".** The model was trained with `class_weight="balanced"`,
which pushes scores towards the positive class, and it was never calibrated. Its scores therefore
spread from about 0.000001 to 0.999999, and the decision threshold sits at 0.81, while the
actual death rate is about 5%. A score of 0.73 does **not** mean a 73% chance of death. Display it
as a score, compared with the threshold, and not as a percentage. Probability wording would need a
calibration step, which has not been done.

> **Proof of concept only.** The model was trained on the eICU *Demo* data, has no external
> validation, and is less accurate for stays shorter than 24h (see the evaluation notes above). It
> is not a clinical decision-making system and must not be used to guide patient care.

## Feature adapter (`feature_adapter.py`)

The UI should not ask anyone for 126 engineered features. `build_model_features(clinical_input)`
converts a small dict of readable clinical fields into the exact record that `predict_mortality()`
expects. `predict_from_clinical()` does both steps in one call.

```python
from ml.feature_adapter import build_model_features, predict_from_clinical, ClinicalInputError
clinical = {"age": 67, "gender": "Male", "unittype": "MICU", "unitstaytype": "admit",
            "intubated": False, "vent": True, "dialysis": False,
            "heartrate": [88, 102, 95], "sao2": 94, "noninvasivesystolic": [110, 96],
            "creatinine": [1.4, 1.9], "lactate": 3.1}
result = predict_from_clinical(clinical)
```

**Fields.** `unittype`, `unitstaytype`, `intubated`, `vent` and `dialysis` are required, because
the model has no "missing" encoding for them. Every other field is optional.

| Field | Value | Model features |
|---|---|---|
| `age` | years; > 89 → 90 (also accepts `"> 89"`), matching eICU de-identification | `age`, `age_missing` |
| `admissionweight` / `admissionheight` | kg / cm | `<field>`, `<field>_missing` |
| `gender` | `Female`, `Male` or None | `gender_female`, `gender_male`, `gender_missing` |
| `unittype` | `Cardiac ICU`, `CCU-CTICU`, `CSICU`, `CTICU`, `Med-Surg ICU`, `MICU`, `Neuro ICU`, `SICU` | 8 `unittype_*` one-hot columns |
| `unitstaytype` | `admit`, `readmit`, `transfer` | 4 `unitstaytype_*` columns (`stepdown_other` is always 0) |
| `heartrate`, `sao2`, `respiration`, `temperature` (°C), `systemicsystolic`, `systemicdiastolic`, `systemicmean` (arterial line), `noninvasivesystolic`, `noninvasivediastolic`, `noninvasivemean` (cuff) | one reading or a list, first 24h of the ICU stay | `<v>_mean`, `_min`, `_max`, `_std`, `_missing` |
| `glucose`, `sodium`, `potassium`, `chloride`, `bicarbonate`, `creatinine`, `bun`, `wbc`, `hemoglobin`, `platelets`, `calcium`, `magnesium`, `lactate` | one result or a chronological list, −6h to +24h | `<lab>_first`, `_min`, `_max`, `_missing` |
| `intubated`, `vent`, `dialysis` | True/False (APACHE first-day flags) | `apache_<field>` |

Units are the eICU units used in training: glucose, creatinine, BUN, calcium and magnesium in
mg/dL; sodium, potassium, chloride, bicarbonate and lactate in mmol/L; haemoglobin in g/dL; WBC
and platelets in ×1000/µL. Category names are case-insensitive.

**How readings become features.** Vitals are summarised by `preprocessing._aggregate_vitals`, the
same function used in training. Labs use the same `first/min/max` aggregation. One-hot columns use
the training category lists. A single reading therefore gives min = max = mean, with a blank std,
which is exactly how a training stay with one observation looks. Tests confirm that feeding the
cleaned raw eICU readings of every training stay through the adapter reproduces
`mortality_training.csv` exactly (all 1,832 stays that have any vital or lab value).

**Missing measurements.** A field that is omitted, None or `[]` is left blank and its
`<var>_missing` flag is set to 1. No value is invented. The saved pipeline imputes the training
median later, exactly as in training. A record with no vitals and no labs at all is rejected.

**Rejected input** (`ClinicalInputError`, a subclass of `MortalityInputError`, which lists every
problem at once):
- unknown field names
- missing required fields
- unsupported categories, including `stepdown/other`, which never occurs in the training cohort
- non-numeric, infinite, NaN or boolean readings
- values outside the cleaning plausibility ranges (temperatures in °F get a hint to use °C)

The adapter rejects such readings, whereas training cleaning set implausible readings to NaN.
Showing the error is safer than silently dropping what someone typed.

**Single readings vs. 24h of readings.** The model learned from 24h summaries. When the held-out
test patients are entered with only their first reading of each vital and lab, ROC-AUC falls from
0.815 to 0.760. Scores also shift downwards: only 7 patients are flagged (5 deaths) instead of 22
(7 deaths), because the threshold was tuned on 24h summaries. Enter all available readings from
the window where possible, and treat snapshot-only scores with extra caution.
