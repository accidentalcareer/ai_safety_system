"""Train and compare ICU-mortality models on the eICU demo patient-level dataset.

Run:  python ml/train.py
(after `python ml/preprocessing.py` has produced ml/data/processed/mortality_training.csv)

Proof of concept for a college project - NOT clinically validated, NOT for patient care.

Protocol
* Patient-grouped, stratified 80/20 split on `uniquepid` (a patient never appears in both sets).
* Every learned preprocessing step lives inside an sklearn Pipeline, fit on training data only.
* Class imbalance: class weights / scale_pos_weight computed from the training fold being fit.
* Model selection uses grouped 5-fold cross-validation on the TRAINING set only; the test set
  is used once, for the final report.
* Decision threshold = F1-maximising threshold on out-of-fold training predictions.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
from preprocessing import (  # noqa: E402
    GROUP_ID, ID_COLUMNS, KEY, LEAKAGE_COLUMNS, LEAKAGE_PATTERNS, TARGET, feature_columns,
)

SEED = 42
N_SPLITS = 5  # 1 of 5 folds held out -> ~20% test
MODEL_VERSION = "0.1.0"
DISPLAY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
           "xgboost": "XGBoost"}
SELECTION_METRICS = ["pr_auc", "roc_auc", "recall", "precision", "f1"]


def build_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """Fresh, unfitted pipelines. Imbalance weights come from `y_train` only."""
    n_pos = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos) / n_pos
    return {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("drop_constant", VarianceThreshold(0.0)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000,
                                         random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("drop_constant", VarianceThreshold(0.0)),
            ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                             class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=SEED)),
        ]),
        # XGBoost handles NaN natively (learns a default branch direction); no imputation.
        "xgboost": Pipeline([
            ("drop_constant", VarianceThreshold(0.0)),
            ("model", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                                    scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                                    tree_method="hist", n_jobs=1, random_state=SEED)),
        ]),
    }


def grouped_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return next(sgkf.split(data, data[TARGET], groups=data[GROUP_ID]))


def check_leakage(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> None:
    shared_patients = set(train[GROUP_ID]) & set(test[GROUP_ID])
    shared_stays = set(train[KEY]) & set(test[KEY])
    assert not shared_patients, f"{len(shared_patients)} patients in both train and test"
    assert not shared_stays, f"{len(shared_stays)} stays in both train and test"
    forbidden = set(ID_COLUMNS) | {TARGET, "actualicumortality"} | LEAKAGE_COLUMNS
    assert not forbidden & set(features), f"forbidden columns in X: {forbidden & set(features)}"
    assert not [c for c in features if LEAKAGE_PATTERNS.search(c)], "leakage-pattern column in X"
    print("Leakage checks passed: 0 shared patients, 0 shared stays, no id/target/outcome columns in X.")


def cross_validate(train: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    """Grouped stratified CV on the training set -> out-of-fold probabilities per model."""
    oof = {name: np.zeros(len(train)) for name in DISPLAY}
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in sgkf.split(train, train[TARGET], groups=train[GROUP_ID]):
        X_tr, y_tr = train.iloc[tr_idx][features], train.iloc[tr_idx][TARGET]
        X_va = train.iloc[va_idx][features]
        for name, pipe in build_models(y_tr).items():  # weights from this fold's training part
            oof[name][va_idx] = pipe.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    return oof


def select_model(cv_metrics: pd.DataFrame) -> str:
    """Lowest mean rank across PR-AUC, ROC-AUC, recall, precision, F1 (CV, training set only).
    Ties are broken by CV PR-AUC."""
    ranks = cv_metrics[SELECTION_METRICS].rank(ascending=False).mean(axis=1)
    order = pd.DataFrame({"rank": ranks, "pr_auc": -cv_metrics["pr_auc"]}).sort_values(["rank", "pr_auc"])
    print("\nSelection (mean rank across CV " + ", ".join(SELECTION_METRICS) + "; lower is better):")
    print(ranks.round(2).to_string())
    return order.index[0]


def fmt(df: pd.DataFrame) -> str:
    cols = ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"]
    return df[cols].rename(index=DISPLAY).round(3).to_string()


def main() -> None:
    data = pd.read_csv(ev.DATA_PATH)
    features = [c for c in feature_columns(data)]

    print(f"Target column: {TARGET} (derived from apachePatientResult.actualicumortality: "
          f"ALIVE -> 0, EXPIRED -> 1)")
    print(f"Target values: {sorted(data[TARGET].unique())}  counts: {data[TARGET].value_counts().to_dict()}  "
          f"positive: {100 * data[TARGET].mean():.2f}%")

    tr_idx, te_idx = grouped_split(data)
    train, test = data.iloc[tr_idx].reset_index(drop=True), data.iloc[te_idx].reset_index(drop=True)
    check_leakage(train, test, features)
    for name, part in [("train", train), ("test", test)]:
        print(f"{name:<5}: {len(part)} stays ({100 * len(part) / len(data):.1f}%), "
              f"{part[GROUP_ID].nunique()} patients, {int(part[TARGET].sum())} deaths "
              f"({100 * part[TARGET].mean():.2f}%)")
    n_pos = int(train[TARGET].sum())
    print(f"Training class ratio negative:positive = {len(train) - n_pos}:{n_pos} "
          f"= {(len(train) - n_pos) / n_pos:.2f}:1 (used for class weights / scale_pos_weight)")

    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    # ---- model selection on training data only
    print(f"\nRunning grouped {N_SPLITS}-fold CV on the training set...")
    oof = cross_validate(train, features)
    thresholds = {name: ev.best_f1_threshold(y_train, p) for name, p in oof.items()}
    cv_metrics = pd.DataFrame({name: ev.compute_metrics(y_train, p, thresholds[name])
                               for name, p in oof.items()}).T
    print("\nCross-validated (out-of-fold) metrics on TRAINING set:")
    print(fmt(cv_metrics))
    best = select_model(cv_metrics)
    print(f"Selected model: {DISPLAY[best]}")

    # ---- final fit on the full training set, single evaluation on the untouched test set
    fitted, test_probs, test_rows, test_rows_05 = {}, {}, {}, {}
    for name, pipe in build_models(y_train).items():
        fitted[name] = pipe.fit(X_train, y_train)
        test_probs[name] = pipe.predict_proba(X_test)[:, 1]
        test_rows[name] = ev.compute_metrics(y_test, test_probs[name], thresholds[name])
        test_rows_05[name] = ev.compute_metrics(y_test, test_probs[name], 0.5)
    test_metrics = pd.DataFrame(test_rows).T
    print("\nTEST set metrics (threshold chosen on training OOF predictions):")
    print(fmt(test_metrics))
    print("\nTEST set metrics at the default 0.5 threshold (reference):")
    print(fmt(pd.DataFrame(test_rows_05).T))
    cis = {name: ev.bootstrap_ci(y_test, p, seed=SEED) for name, p in test_probs.items()}
    print("\nTest-set bootstrap 95% CIs (1000 resamples):")
    for name, ci in cis.items():
        print(f"  {DISPLAY[name]:<20} ROC-AUC [{ci['roc_auc'][0]:.3f}, {ci['roc_auc'][1]:.3f}]  "
              f"PR-AUC [{ci['pr_auc'][0]:.3f}, {ci['pr_auc'][1]:.3f}]")

    # ---- artefacts
    ev.MODELS_DIR.mkdir(exist_ok=True)
    ev.RESULTS_DIR.mkdir(exist_ok=True)
    named_probs = {DISPLAY[n]: p for n, p in test_probs.items()}
    ev.plot_roc(y_test, named_probs, ev.RESULTS_DIR / "roc_curve.png")
    ev.plot_pr(y_test, named_probs, ev.RESULTS_DIR / "pr_curve.png")
    ev.plot_confusion(y_test, test_probs[best], thresholds[best], DISPLAY[best],
                      ev.RESULTS_DIR / "confusion_matrix_best.png")

    pd.concat([cv_metrics.assign(evaluation="cv_train_oof"),
               test_metrics.assign(evaluation="test"),
               pd.DataFrame(test_rows_05).T.assign(evaluation="test_threshold_0.5")]) \
        .rename_axis("model").reset_index().to_csv(ev.RESULTS_DIR / "metrics.csv", index=False)
    pd.concat([train[[KEY, GROUP_ID]].assign(split="train"), test[[KEY, GROUP_ID]].assign(split="test")]) \
        .to_csv(ev.SPLIT_PATH, index=False)

    joblib.dump(fitted[best], ev.MODEL_PATH)
    kept = fitted[best].named_steps["drop_constant"].get_support()
    metadata = {
        "model_name": DISPLAY[best],
        "model_key": best,
        "model_version": MODEL_VERSION,
        "created": date.today().isoformat(),
        "disclaimer": ("College-project proof of concept trained on the eICU Collaborative Research "
                       "Database Demo. NOT clinically validated and NOT for patient-care decisions. "
                       "Output is a demonstration risk estimate, not a diagnosis."),
        "target": TARGET,
        "target_definition": "ICU mortality from apachePatientResult.actualicumortality "
                             "(ALIVE=0, EXPIRED=1); not an APACHE prediction",
        "feature_names": features,
        "features_dropped_by_variance_filter": [f for f, k in zip(features, kept) if not k],
        "n_features_input": len(features),
        "observation_window": "vitals 0-24h after ICU admission; labs -6h to +24h",
        "decision_threshold": thresholds[best],
        "decision_threshold_method": "F1-maximising threshold on grouped 5-fold out-of-fold "
                                     "predictions on the training set",
        "random_seed": SEED,
        "split": {"method": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True), first fold "
                            f"held out", "group_column": GROUP_ID,
                  "train_size": len(train), "test_size": len(test),
                  "train_patients": int(train[GROUP_ID].nunique()),
                  "test_patients": int(test[GROUP_ID].nunique()),
                  "train_positives": n_pos, "test_positives": int(y_test.sum()),
                  "split_file": "ml/results/split.csv"},
        "class_imbalance": {"train_negative_to_positive_ratio": (len(train) - n_pos) / n_pos,
                            "logistic_regression": "class_weight='balanced'",
                            "random_forest": "class_weight='balanced_subsample'",
                            "xgboost": f"scale_pos_weight={(len(train) - n_pos) / n_pos:.4f}"},
        "preprocessing": {
            "upstream": "ml/preprocessing.py: 24h aggregation, plausibility cleaning, one-hot "
                        "encoding with fixed categories, <var>_missing indicators",
            "pipeline_steps": [s for s, _ in fitted[best].steps],
            "logistic_regression": "median imputation -> drop constant features -> standard scaling",
            "random_forest": "median imputation -> drop constant features",
            "xgboost": "drop constant features; NaN handled natively",
            "fit_on": "training split only",
        },
        "selection": {"criterion": "lowest mean rank across CV " + ", ".join(SELECTION_METRICS)
                                   + " on the training set (tie-break: CV PR-AUC)",
                      "cv_metrics": {DISPLAY[n]: {k: float(v) for k, v in r.items()}
                                     for n, r in cv_metrics.iterrows()}},
        "test_metrics": {k: float(v) for k, v in test_rows[best].items()},
        "test_metrics_all_models": {DISPLAY[n]: {k: float(v) for k, v in r.items()}
                                    for n, r in test_rows.items()},
        "test_bootstrap_95ci": cis[best],
        "library_versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__},
    }
    ev.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model    -> {ev.MODEL_PATH}")
    print(f"Saved metadata -> {ev.METADATA_PATH}")
    print(f"Saved results  -> {ev.RESULTS_DIR}")


if __name__ == "__main__":
    main()
