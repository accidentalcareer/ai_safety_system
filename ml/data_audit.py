"""Read-only audit of the raw eICU demo tables used by the mortality pipeline.

Run:  python ml/data_audit.py
Nothing is written; the raw files are only read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
KEY = "patientunitstayid"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 300)


def read_raw(stem: str, **kwargs) -> pd.DataFrame:
    for suffix in (".csv", ".csv.gz"):
        path = RAW_DIR / f"{stem}{suffix}"
        if path.exists():
            return pd.read_csv(path, low_memory=False, **kwargs)
    raise FileNotFoundError(f"{stem}.csv(.gz) not found in {RAW_DIR}")


def header(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def overview(name: str, df: pd.DataFrame, relevant: list[str]) -> None:
    header(name)
    print(f"rows={len(df):,}  cols={df.shape[1]}")
    print(f"columns: {list(df.columns)}")
    print(f"unique {KEY}: {df[KEY].nunique():,}")
    dup_keys = df[KEY].duplicated().sum()
    print(f"rows sharing a {KEY} with an earlier row: {dup_keys:,}")
    print(f"fully duplicated rows: {df.duplicated().sum():,}")
    cols = [c for c in relevant if c in df.columns]
    info = pd.DataFrame(
        {
            "dtype": df[cols].dtypes.astype(str),
            "missing_%": df[cols].isna().mean().mul(100).round(2),
            "n_minus1": [(df[c] == -1).sum() if pd.api.types.is_numeric_dtype(df[c]) else 0 for c in cols],
        }
    )
    print(info)


def numeric_ranges(df: pd.DataFrame, cols: list[str]) -> None:
    desc = df[cols].describe(percentiles=[0.01, 0.5, 0.99]).T
    print(desc[["count", "min", "1%", "50%", "99%", "max"]].round(2))


def main() -> None:
    # ---------------- patient ----------------
    patient = read_raw("patient")
    p_cols = ["age", "gender", "admissionweight", "admissionheight", "unittype", "unitstaytype",
              "unitdischargestatus", "hospitaldischargestatus", "unitdischargeoffset"]
    overview("patient.csv", patient, p_cols)
    for c in ["gender", "unittype", "unitstaytype", "unitdischargestatus"]:
        print(f"\n{c}:\n{patient[c].value_counts(dropna=False)}")
    age_num = pd.to_numeric(patient["age"], errors="coerce")
    print(f"\nnon-numeric age values: {patient.loc[age_num.isna(), 'age'].value_counts(dropna=False).to_dict()}")
    print(f"numeric age range: {age_num.min()} - {age_num.max()}")
    numeric_ranges(patient, ["admissionweight", "admissionheight", "unitdischargeoffset"])
    print(f"stays per uniquepid: max={patient.groupby('uniquepid').size().max()}, "
          f"patients with >1 stay={int((patient.groupby('uniquepid').size() > 1).sum())}")

    # ---------------- vitalPeriodic ----------------
    vp_cols = ["heartrate", "sao2", "respiration", "temperature",
               "systemicsystolic", "systemicdiastolic", "systemicmean"]
    vp = read_raw("vitalPeriodic", usecols=[KEY, "observationoffset", *vp_cols])
    overview("vitalPeriodic.csv (selected columns)", vp, ["observationoffset", *vp_cols])
    numeric_ranges(vp, ["observationoffset", *vp_cols])
    print(f"rows with observationoffset < 0: {(vp.observationoffset < 0).sum():,}")
    print(f"rows within first 24h [0,1440]: {vp.observationoffset.between(0, 1440).sum():,}")
    print(f"duplicate ({KEY}, observationoffset) rows: {vp.duplicated([KEY, 'observationoffset']).sum():,}")

    # ---------------- vitalAperiodic ----------------
    va_cols = ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"]
    va = read_raw("vitalAperiodic", usecols=[KEY, "observationoffset", *va_cols])
    overview("vitalAperiodic.csv (selected columns)", va, ["observationoffset", *va_cols])
    numeric_ranges(va, ["observationoffset", *va_cols])
    print(f"rows with observationoffset < 0: {(va.observationoffset < 0).sum():,}")

    # ---------------- lab ----------------
    lab = read_raw("lab")
    overview("lab.csv", lab, ["labresultoffset", "labname", "labresult", "labresulttext", "labmeasurenamesystem"])
    counts = lab.groupby("labname").agg(n_obs=("labresult", "size"), n_stays=(KEY, "nunique"),
                                         unit=("labmeasurenamesystem", lambda s: s.dropna().unique()[:3].tolist()))
    print("\nALL unique labname values:")
    print(counts.sort_index())
    wanted = ["glucose", "sodium", "potassium", "chloride", "bicarbonate", "creatinine", "bun",
              "wbc", "hemoglobin", "platelet", "calcium", "magnesium", "lactate", "hgb", "hco3", "co2"]
    print("\nCandidate labnames for requested labs (substring match, case-insensitive):")
    for w in wanted:
        hits = [n for n in counts.index if w in n.lower()]
        print(f"  {w:<12} -> {hits}")
    print(f"\nlabresult NaN but labresulttext present: "
          f"{(lab.labresult.isna() & lab.labresulttext.notna()).sum():,}")
    print(f"labs with labresultoffset < 0: {(lab.labresultoffset < 0).sum():,}")

    # ---------------- apacheApsVar ----------------
    aps = read_raw("apacheApsVar")
    overview("apacheApsVar.csv", aps, list(aps.columns[2:]))
    for c in ["intubated", "vent", "dialysis"]:
        print(f"{c}: {aps[c].value_counts(dropna=False).sort_index().to_dict()}")

    # ---------------- apachePatientResult ----------------
    apr = read_raw("apachePatientResult")
    overview("apachePatientResult.csv", apr, ["apacheversion", "actualicumortality", "actualhospitalmortality",
                                              "predictedicumortality", "actualiculos"])
    print(f"\nactualicumortality value_counts (incl. NaN):\n{apr.actualicumortality.value_counts(dropna=False)}")
    print(f"\nby apacheversion:\n{pd.crosstab(apr.apacheversion, apr.actualicumortality, dropna=False)}")
    per_stay = apr.groupby(KEY).actualicumortality.nunique(dropna=False)
    print(f"stays whose rows disagree on actualicumortality: {(per_stay > 1).sum()}")
    print(f"rows per stay: {apr.groupby(KEY).size().value_counts().to_dict()}")
    print(f"predictedicumortality == -1 rows: {(apr.predictedicumortality == -1).sum()}")

    one = apr.drop_duplicates(KEY)[[KEY, "actualicumortality"]]
    merged = one.merge(patient[[KEY, "unitdischargestatus"]], on=KEY, how="left")
    print(f"\nactualicumortality vs patient.unitdischargestatus:\n"
          f"{pd.crosstab(merged.actualicumortality, merged.unitdischargestatus, dropna=False)}")
    print(f"\npatient stays without an apachePatientResult row: {(~patient[KEY].isin(apr[KEY])).sum()}")
    print(f"apache result stays without apacheApsVar row: {(~one[KEY].isin(aps[KEY])).sum()}")

    # coverage of time-series within the cohort
    cohort = set(one[KEY])
    for name, df in [("vitalPeriodic", vp), ("vitalAperiodic", va), ("lab", lab)]:
        print(f"cohort stays with any {name} row: {len(cohort & set(df[KEY])):,} / {len(cohort):,}")

    # how many cohort deaths happen within the first 24h (relevant to window choice)
    los = one.merge(patient[[KEY, "unitdischargeoffset"]], on=KEY)
    died = los[los.actualicumortality == "EXPIRED"]
    if len(died):
        print(f"\ndeaths: {len(died)}; died within 24h of unit admit: "
              f"{(died.unitdischargeoffset <= 1440).sum()}; median offset (h): "
              f"{np.median(died.unitdischargeoffset) / 60:.1f}")


if __name__ == "__main__":
    main()
