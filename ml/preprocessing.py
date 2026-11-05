"""Build a patient-level (one row per ICU stay) ICU-mortality dataset from the eICU demo.

Run:  python ml/preprocessing.py
Output: ml/data/processed/mortality_training.csv

Design decisions (see ml/README.md for the full rationale):
* Cohort: stays with an `apachePatientResult` row; target = actualicumortality (EXPIRED=1, ALIVE=0).
* Observation window: first 24h after unit admission for vitals, -6h..+24h for labs.
* Physiologically impossible values become NaN (data validity only - never used to build labels).
* Missing values are kept as NaN in the CSV, with a `<var>_missing` indicator per source variable.
  Median imputation is applied later, inside the training pipeline, fit on the training split only
  (see `build_imputer`), so test-set statistics never leak into training.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
RAW_DIR = ML_DIR / "data" / "raw"
PROCESSED_DIR = ML_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "mortality_training.csv"

KEY = "patientunitstayid"
GROUP_ID = "uniquepid"  # patient identifier: kept for grouped train/test splitting, never a feature
TARGET = "icu_mortality"
ID_COLUMNS = [KEY, GROUP_ID]

# Offsets are minutes relative to unit admission.
VITAL_WINDOW = (0, 24 * 60)
LAB_WINDOW = (-6 * 60, 24 * 60)

TARGET_MAPPING = {"ALIVE": 0, "EXPIRED": 1}

# Fixed category lists (observed in the audit) so the feature schema is stable at inference time.
GENDER_CATEGORIES = ["Female", "Male"]
UNITTYPE_CATEGORIES = ["Cardiac ICU", "CCU-CTICU", "CSICU", "CTICU", "Med-Surg ICU", "MICU",
                       "Neuro ICU", "SICU"]
UNITSTAYTYPE_CATEGORIES = ["admit", "readmit", "stepdown/other", "transfer"]

PERIODIC_VITALS = ["heartrate", "sao2", "respiration", "temperature",
                   "systemicsystolic", "systemicdiastolic", "systemicmean"]
APERIODIC_VITALS = ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"]
VITAL_STATS = ["mean", "min", "max", "std"]

# feature name -> exact eICU labname (verified against the unique labname values in the audit)
LAB_NAMES = {
    "glucose": "glucose",              # serum glucose; 'bedside glucose' (fingerstick) is a different assay
    "sodium": "sodium",
    "potassium": "potassium",
    "chloride": "chloride",
    "bicarbonate": "bicarbonate",      # serum chemistry; 'HCO3' (ABG-derived) and 'Total CO2' excluded
    "creatinine": "creatinine",        # serum; 'urinary creatinine' excluded
    "bun": "BUN",
    "wbc": "WBC x 1000",
    "hemoglobin": "Hgb",
    "platelets": "platelets x 1000",
    "calcium": "calcium",              # total calcium; 'ionized calcium' excluded
    "magnesium": "magnesium",
    "lactate": "lactate",
}
LAB_STATS = ["first", "min", "max"]

APACHE_FLAGS = ["intubated", "vent", "dialysis"]

# Broad physiological validity ranges (inclusive). Values outside become NaN.
PLAUSIBLE_RANGES = {
    "age": (0, 120),
    "admissionweight": (20, 350),        # kg
    "admissionheight": (100, 250),       # cm
    "heartrate": (20, 300),
    "sao2": (20, 100),
    "respiration": (1, 100),
    "temperature": (25, 45),             # degC, after F->C conversion
    "systemicsystolic": (20, 300),
    "systemicdiastolic": (5, 250),
    "systemicmean": (10, 250),
    "noninvasivesystolic": (20, 300),
    "noninvasivediastolic": (5, 250),
    "noninvasivemean": (10, 250),
    "glucose": (5, 3000),
    "sodium": (80, 200),
    "potassium": (1, 15),
    "chloride": (40, 180),
    "bicarbonate": (1, 80),
    "creatinine": (0.05, 40),
    "bun": (1, 300),
    "wbc": (0.01, 1000),
    "hemoglobin": (1, 25),
    "platelets": (1, 3000),
    "calcium": (2, 25),
    "magnesium": (0.2, 15),
    "lactate": (0.1, 40),
}
FAHRENHEIT_RANGE = (80, 115)  # temperatures recorded in degF in a degC column

# Columns that encode the outcome, a prediction of it, or post-outcome information.
LEAKAGE_COLUMNS = {
    "actualicumortality", "actualhospitalmortality",
    "predictedicumortality", "predictedhospitalmortality",
    "predictediculos", "predictedhospitallos", "predventdays",
    "actualiculos", "actualhospitallos", "actualventdays", "unabridgedactualventdays",
    "unabridgedunitlos", "unabridgedhosplos", "apachescore", "acutephysiologyscore",
    "unitdischargestatus", "hospitaldischargestatus", "unitdischargeoffset",
    "hospitaldischargeoffset", "unitdischargelocation", "hospitaldischargelocation",
    "dischargeweight", "unitdischargetime24", "hospitaldischargetime24", "hospitaldischargeyear",
}
LEAKAGE_PATTERNS = re.compile(r"predicted|actual|unabridged|discharge|apachescore|acutephysiology|mortality")


def read_raw(stem: str, **kwargs) -> pd.DataFrame:
    """Read `<stem>.csv` or `<stem>.csv.gz` from the raw directory (read-only)."""
    for suffix in (".csv", ".csv.gz"):
        path = RAW_DIR / f"{stem}{suffix}"
        if path.exists():
            return pd.read_csv(path, low_memory=False, **kwargs)
    raise FileNotFoundError(f"{stem}.csv(.gz) not found in {RAW_DIR}")


def mask_implausible(df: pd.DataFrame, columns: list[str], log: dict[str, int]) -> pd.DataFrame:
    for col in columns:
        low, high = PLAUSIBLE_RANGES[col]
        bad = df[col].notna() & ~df[col].between(low, high)
        log[col] = log.get(col, 0) + int(bad.sum())
        df.loc[bad, col] = np.nan
    return df


# --------------------------------------------------------------------------- target / cohort

def load_target() -> pd.DataFrame:
    apr = read_raw("apachePatientResult", usecols=[KEY, "apacheversion", "actualicumortality"])
    values = set(apr["actualicumortality"].dropna().unique())
    if values - TARGET_MAPPING.keys():
        raise ValueError(f"Unexpected actualicumortality values: {values - TARGET_MAPPING.keys()}")
    # Each stay has an APACHE IV and an IVa row; the outcome must agree between them.
    disagreeing = apr.groupby(KEY)["actualicumortality"].nunique(dropna=False).gt(1).sum()
    if disagreeing:
        raise ValueError(f"{disagreeing} stays have conflicting actualicumortality across APACHE versions")
    target = apr.drop_duplicates(KEY)[[KEY, "actualicumortality"]].dropna()
    target[TARGET] = target.pop("actualicumortality").map(TARGET_MAPPING).astype(int)
    return target


# --------------------------------------------------------------------------- static features

def build_patient_features(cohort: pd.Index, clean_log: dict[str, int]) -> pd.DataFrame:
    cols = [KEY, GROUP_ID, "age", "gender", "admissionweight", "admissionheight", "unittype", "unitstaytype"]
    patient = read_raw("patient", usecols=cols)
    patient = patient[patient[KEY].isin(cohort)].copy()

    # eICU de-identifies ages above 89 as the string "> 89"; map to 90.
    patient["age"] = pd.to_numeric(patient["age"].replace({"> 89": "90"}), errors="coerce")
    patient = mask_implausible(patient, ["age", "admissionweight", "admissionheight"], clean_log)

    out = patient[[KEY, GROUP_ID, "age", "admissionweight", "admissionheight"]].copy()
    for col in ["age", "admissionweight", "admissionheight"]:
        out[f"{col}_missing"] = out[col].isna().astype(int)

    # One-hot with fixed categories; an unknown/missing category is all zeros.
    for col, categories in [("gender", GENDER_CATEGORIES), ("unittype", UNITTYPE_CATEGORIES),
                            ("unitstaytype", UNITSTAYTYPE_CATEGORIES)]:
        unseen = set(patient[col].dropna().unique()) - set(categories)
        if unseen:
            raise ValueError(f"Unseen {col} categories: {unseen}")
        for cat in categories:
            name = f"{col}_{re.sub(r'[^0-9a-zA-Z]+', '_', cat).strip('_').lower()}"
            out[name] = (patient[col] == cat).astype(int)
    out["gender_missing"] = patient["gender"].isna().astype(int).values
    return out


def build_apache_flags(cohort: pd.Index) -> pd.DataFrame:
    aps = read_raw("apacheApsVar", usecols=[KEY, *APACHE_FLAGS])
    aps = aps[aps[KEY].isin(cohort)].drop_duplicates(KEY)
    for col in APACHE_FLAGS:
        # The audit found no -1 sentinels in these three flags; enforce it.
        bad = set(aps[col].dropna().unique()) - {0, 1}
        if bad:
            raise ValueError(f"apacheApsVar.{col} has non-binary values {bad}")
    return aps.rename(columns={c: f"apache_{c}" for c in APACHE_FLAGS})


# --------------------------------------------------------------------------- time series

def _aggregate_vitals(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    agg = df.groupby(KEY)[variables].agg(VITAL_STATS)
    agg.columns = [f"{var}_{stat}" for var, stat in agg.columns]
    counts = df.groupby(KEY)[variables].count()
    for var in variables:
        agg[f"{var}_missing"] = (counts[var] == 0).astype(int)
    return agg


def build_vital_features(cohort: pd.Index, clean_log: dict[str, int]) -> pd.DataFrame:
    lo, hi = VITAL_WINDOW

    vp = read_raw("vitalPeriodic", usecols=[KEY, "observationoffset", *PERIODIC_VITALS])
    vp = vp[vp[KEY].isin(cohort) & vp["observationoffset"].between(lo, hi)].drop_duplicates()
    fahrenheit = vp["temperature"].between(*FAHRENHEIT_RANGE)
    clean_log["temperature_converted_from_F"] = int(fahrenheit.sum())
    vp.loc[fahrenheit, "temperature"] = (vp.loc[fahrenheit, "temperature"] - 32) * 5 / 9
    vp = mask_implausible(vp, PERIODIC_VITALS, clean_log)

    va = read_raw("vitalAperiodic", usecols=[KEY, "observationoffset", *APERIODIC_VITALS])
    va = va[va[KEY].isin(cohort) & va["observationoffset"].between(lo, hi)].drop_duplicates()
    va = mask_implausible(va, APERIODIC_VITALS, clean_log)

    frames = [_aggregate_vitals(vp, PERIODIC_VITALS), _aggregate_vitals(va, APERIODIC_VITALS)]
    return pd.concat(frames, axis=1).reset_index()


def _parse_censored(text: pd.Series) -> pd.Series:
    """'<0.50' / '>40' style results (no numeric labresult) -> their numeric bound."""
    return pd.to_numeric(text.str.replace(r"^\s*[<>]=?\s*", "", regex=True), errors="coerce")


def build_lab_features(cohort: pd.Index, clean_log: dict[str, int]) -> pd.DataFrame:
    lo, hi = LAB_WINDOW
    cols = [KEY, "labresultoffset", "labresultrevisedoffset", "labname", "labresult", "labresulttext"]
    lab = read_raw("lab", usecols=cols)
    name_to_feature = {v: k for k, v in LAB_NAMES.items()}
    lab = lab[lab[KEY].isin(cohort) & lab["labname"].isin(name_to_feature)
              & lab["labresultoffset"].between(lo, hi)].copy()
    lab["feature"] = lab["labname"].map(name_to_feature)

    censored = lab["labresult"].isna()
    lab.loc[censored, "labresult"] = _parse_censored(lab.loc[censored, "labresulttext"])
    clean_log["lab_censored_values_parsed"] = int((censored & lab["labresult"].notna()).sum())
    lab = lab.dropna(subset=["labresult"])

    # Same lab at the same time reported more than once: keep the latest revision.
    lab = (lab.sort_values("labresultrevisedoffset")
              .drop_duplicates([KEY, "labresultoffset", "feature"], keep="last"))

    for feature, (low, high) in PLAUSIBLE_RANGES.items():
        if feature in LAB_NAMES:
            bad = (lab["feature"] == feature) & ~lab["labresult"].between(low, high)
            clean_log[feature] = int(bad.sum())
            lab = lab[~bad]

    lab = lab.sort_values([KEY, "feature", "labresultoffset"])
    agg = lab.groupby([KEY, "feature"])["labresult"].agg(LAB_STATS).unstack("feature")
    agg.columns = [f"{feature}_{stat}" for stat, feature in agg.columns]
    ordered = [f"{f}_{s}" for f in LAB_NAMES for s in LAB_STATS]
    agg = agg.reindex(columns=ordered)
    for feature in LAB_NAMES:
        agg[f"{feature}_missing"] = agg[f"{feature}_first"].isna().astype(int)
    return agg.reset_index()


# --------------------------------------------------------------------------- assembly / checks

def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS and c != TARGET]


def build_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    clean_log: dict[str, int] = {}
    target = load_target()
    cohort = pd.Index(target[KEY])

    data = (target
            .merge(build_patient_features(cohort, clean_log), on=KEY, how="left")
            .merge(build_apache_flags(cohort), on=KEY, how="left")
            .merge(build_vital_features(cohort, clean_log), on=KEY, how="left")
            .merge(build_lab_features(cohort, clean_log), on=KEY, how="left"))

    # Stays with no rows at all in a time-series table get indicator = 1.
    for col in [c for c in data.columns if c.endswith("_missing")]:
        data[col] = data[col].fillna(1).astype(int)

    data = data.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    data = data[[*ID_COLUMNS, TARGET, *feature_columns(data)]]
    return data, clean_log


def validate(data: pd.DataFrame) -> None:
    features = feature_columns(data)
    assert data[KEY].is_unique, "patientunitstayid is not unique"
    assert data[TARGET].notna().all(), "target missing for some rows"
    assert set(data[TARGET].unique()) <= {0, 1}, "target is not binary 0/1"
    assert TARGET not in features and "actualicumortality" not in features, "target leaked into features"
    leaked = [c for c in features if c in LEAKAGE_COLUMNS or LEAKAGE_PATTERNS.search(c)]
    assert not leaked, f"leakage columns in features: {leaked}"
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(data[c])]
    assert not non_numeric, f"non-numeric feature columns: {non_numeric}"
    assert not np.isinf(data[features].to_numpy(dtype=float)).any(), "infinite values remain"
    assert not data.duplicated().any(), "duplicate rows remain"


def build_imputer():
    """Median imputer for the numeric features. Fit it on the TRAINING split only.

    Missingness is already preserved by the `<var>_missing` indicator columns.
    """
    from sklearn.impute import SimpleImputer

    return SimpleImputer(strategy="median", keep_empty_features=True)


def main() -> None:
    data, clean_log = build_dataset()
    validate(data)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    features = feature_columns(data)
    positives = int(data[TARGET].sum())
    print(f"Wrote {OUTPUT_PATH}")
    print(f"shape: {data.shape} ({len(features)} features + {len(ID_COLUMNS)} id columns + target)")
    print(f"positive (EXPIRED): {positives}  negative (ALIVE): {len(data) - positives}  "
          f"positive %: {100 * positives / len(data):.2f}")
    print(f"distinct patients ({GROUP_ID}): {data[GROUP_ID].nunique()}")
    print("\nvalues set to NaN / converted during cleaning:")
    for k, v in clean_log.items():
        if v:
            print(f"  {k}: {v}")
    missing = data[features].isna().mean().mul(100).round(1)
    print(f"\nfeatures with missing values: {(missing > 0).sum()} / {len(features)}")
    print(missing[missing > 0].sort_values(ascending=False).to_string())
    print(f"\nfeatures:\n{features}")


if __name__ == "__main__":
    main()
