"""Metrics and ordinal-label helpers for 0-4 diabetic retinopathy grades."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

N_GRADES = 5


def quadratic_kappa(y_true, y_pred) -> float:
    """Quadratic weighted Cohen's kappa (the standard APTOS/EyePACS metric)."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def summarize(y_true, y_pred) -> dict:
    return {
        "qwk": quadratic_kappa(y_true, y_pred),
        "acc": float(accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(N_GRADES))).tolist(),
    }


# --- Ordinal (cumulative-link, CORAL-style) helpers -------------------------
# The head outputs K-1 logits; logit k predicts P(grade > k).

def ordinal_targets(y: torch.Tensor, n_classes: int = N_GRADES) -> torch.Tensor:
    ks = torch.arange(n_classes - 1, device=y.device)
    return (y.unsqueeze(1) > ks).float()


def ordinal_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, ordinal_targets(y, logits.shape[1] + 1))


def ordinal_predict(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def head_outputs(loss: str, n_classes: int = N_GRADES) -> int:
    return n_classes - 1 if loss == "ordinal" else n_classes


def predict(logits: torch.Tensor, loss: str) -> np.ndarray:
    preds = ordinal_predict(logits) if loss == "ordinal" else logits.argmax(dim=1)
    return preds.cpu().numpy()
