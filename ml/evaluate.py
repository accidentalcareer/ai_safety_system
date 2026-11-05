"""Evaluation metrics and plots for the ICU-mortality models.

Imported by train.py. Run on its own to re-evaluate the saved model on the saved test split:
    python ml/evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

ML_DIR = Path(__file__).resolve().parent
DATA_PATH = ML_DIR / "data" / "processed" / "mortality_training.csv"
MODELS_DIR = ML_DIR / "models"
RESULTS_DIR = ML_DIR / "results"
MODEL_PATH = MODELS_DIR / "aegiscare_mortality_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SPLIT_PATH = RESULTS_DIR / "split.csv"

THRESHOLD_FREE = ["roc_auc", "pr_auc"]


def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def best_f1_threshold(y_true, y_prob) -> float:
    """Threshold maximising F1. Call this on training (out-of-fold) predictions only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def bootstrap_ci(y_true, y_prob, n_boot: int = 1000, seed: int = 42) -> dict:
    """95% percentile bootstrap CIs for ROC-AUC and PR-AUC on the test set."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    scores = {"roc_auc": [], "pr_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        scores["roc_auc"].append(roc_auc_score(y_true[idx], y_prob[idx]))
        scores["pr_auc"].append(average_precision_score(y_true[idx], y_prob[idx]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in scores.items()}


def plot_roc(y_true, probs: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC curve - test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(y_true, probs: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        precision, recall, _ = precision_recall_curve(y_true, p)
        ax.step(recall, precision, where="post",
                label=f"{name} (AP = {average_precision_score(y_true, p):.3f})")
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, color="k", ls="--", lw=1, label=f"Prevalence = {prevalence:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve - test set",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_prob, threshold: float, name: str, path: Path) -> None:
    cm = confusion_matrix(y_true, (np.asarray(y_prob) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Survived (0)", "Died (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{name} - test set\n(threshold = {threshold:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Reload the saved pipeline and confirm it reproduces the recorded test metrics."""
    model = joblib.load(MODEL_PATH)
    meta = json.loads(METADATA_PATH.read_text())
    data = pd.read_csv(DATA_PATH)
    split = pd.read_csv(SPLIT_PATH)
    test_ids = split.loc[split["split"] == "test", "patientunitstayid"]
    test = data[data["patientunitstayid"].isin(test_ids)]

    X_test, y_test = test[meta["feature_names"]], test[meta["target"]]
    prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, prob, meta["decision_threshold"])
    recorded = meta["test_metrics"]

    print(f"Model: {meta['model_name']}  (version {meta['model_version']})")
    print(f"Test set: {len(test)} stays, {int(y_test.sum())} deaths")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"]:
        print(f"  {key:<10} {metrics[key]:.4f}" if isinstance(metrics[key], float) else f"  {key:<10} {metrics[key]}")
    mismatched = [k for k in recorded if k in metrics and not np.isclose(recorded[k], metrics[k])]
    print("Reproduces metadata metrics: " + ("YES" if not mismatched else f"NO - differs on {mismatched}"))


if __name__ == "__main__":
    main()
