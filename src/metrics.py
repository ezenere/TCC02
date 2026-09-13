"""Classification metrics in the single format every axis reports.

Every reported number is accuracy, macro F1 and error rate; the error ratio
against a baseline is computed downstream by the results scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_metrics(y_true, y_pred, classes: list[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    acc = float((y_true == y_pred).mean())
    return {
        "n": int(len(y_true)),
        "acc": acc,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "error_rate": 1.0 - acc,
        "n_errors": int((y_true != y_pred).sum()),
        "per_class": [{
            "label": c, "support": int(support[i]),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]),
        } for i, c in enumerate(classes)],
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
    }


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path
