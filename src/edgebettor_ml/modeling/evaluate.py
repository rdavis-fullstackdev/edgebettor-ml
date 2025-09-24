from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics as skm


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(skm.roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return float(skm.log_loss(y_true, y_prob))


