"""Metric helpers matching the existing EIN trainer."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy_score,
    f1_score as sklearn_f1_score,
    roc_auc_score,
)


def classification_metrics(logits_or_prediction, target):
    """Return acc/auc/f1 exactly like trainer/EIN_trainer.py.

    The current project computes AUC from predicted class labels rather than
    class probabilities. TCSR follows that behavior for fair comparison.
    """
    if torch.is_tensor(logits_or_prediction):
        if logits_or_prediction.dim() > 1:
            prediction = logits_or_prediction.argmax(dim=-1)
        else:
            prediction = logits_or_prediction
    else:
        prediction = logits_or_prediction
    y_true = _to_numpy(target).reshape(-1)
    y_pred = _to_numpy(prediction).reshape(-1)

    acc = sklearn_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = np.nan
    f1 = sklearn_f1_score(y_true, y_pred)
    return {
        "acc": float(acc),
        "auc": float(auc),
        "f1": float(f1),
    }


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)
