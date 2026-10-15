"""Metrics and threshold selection used by classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Calculate AUC and threshold-dependent classification metrics."""
    prediction = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose a threshold from training predictions using Youden's J."""
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best_threshold, best_value = 0.5, -np.inf
    for threshold in candidates:
        result = classification_metrics(y_true, scores, float(threshold))
        value = float(result["sensitivity"] + result["specificity"] - 1)
        closer = abs(threshold - 0.5) < abs(best_threshold - 0.5)
        if value > best_value or (value == best_value and closer):
            best_threshold, best_value = float(threshold), value
    return best_threshold

