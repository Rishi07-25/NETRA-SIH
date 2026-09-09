"""Evaluation metric definitions and computation routines for NETRA Threat Engine.

Calculates multiclass classification metrics (macro F1, weighted F1, recall, precision)
and unsupervised anomaly detection evaluation metrics (precision, recall, false positive rate).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, List[str]],
    y_pred: Union[pd.Series, np.ndarray, List[str]],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for multiclass attack attribution.

    Parameters
    ----------
    y_true : Ground truth attack labels.
    y_pred : Model predicted attack classes.
    labels : Optional list of unique class names to order confusion matrix.

    Returns
    -------
    Dict[str, Any]
        Dictionary of accuracy, precision, recall, F1 (macro & weighted),
        confusion matrix, and per-class report.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    prec_macro = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    target_labels = labels if labels is not None else sorted(list(set(y_t) | set(y_p)))
    cm = confusion_matrix(y_t, y_p, labels=target_labels)
    report = classification_report(y_t, y_p, labels=target_labels, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "labels": target_labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def compute_anomaly_metrics(
    y_true_labels: Union[pd.Series, np.ndarray, List[str]],
    is_anomalous_pred: Union[pd.Series, np.ndarray, List[bool]],
    benign_label: str = "BENIGN",
) -> Dict[str, Any]:
    """Evaluate unsupervised anomaly detection using ground truth labels.

    NOTE:
    Labels are utilized strictly for POST-TRAINING EVALUATION.
    They are never exposed to the Isolation Forest during model fitting.

    Binary Mapping:
    - Ground Truth Anomaly: y_true != benign_label (1)
    - Ground Truth Normal:  y_true == benign_label (0)
    - Prediction: is_anomalous_pred (bool) -> 1 if True else 0

    Returns
    -------
    Dict[str, Any]
        Precision, Recall, F1, and False Positive Rate (FPR).
    """
    y_true_binary = np.array([0 if str(lbl).upper() == benign_label.upper() else 1 for lbl in y_true_labels])
    y_pred_binary = np.array([1 if bool(p) else 0 for p in is_anomalous_pred])

    # Confusion components
    # 0 = Normal, 1 = Anomaly
    # tn: normal predicted normal
    # fp: normal predicted anomaly (false alarm)
    # fn: attack predicted normal (miss)
    # tp: attack predicted anomaly (detection)
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "anomaly_precision": round(prec, 4),
        "anomaly_recall": round(rec, 4),
        "anomaly_f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "evaluation_note": (
            "Labels were used strictly for post-training evaluation. "
            "The Isolation Forest model was fitted completely unsupervised on feature data only."
        ),
    }
