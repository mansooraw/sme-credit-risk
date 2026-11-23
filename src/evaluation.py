"""Cost based evaluation for CreditLens (CRISP-DM Evaluation).

The cost matrix of section 7 of the project brief, shared by notebook 05 and
the Streamlit app. All constants come from src/config.py.

For a loan with disbursed amount D and guaranteed share g:

* approved and repaid:   + NET_INTEREST_MARGIN * D
* approved and default:  - (LOSS_GIVEN_DEFAULT * D * (1 - g) + WORKOUT_COST_RATE * D)
* declined:              0

Main entry points
-----------------
gain_if_repaid, loss_if_default   per-loan value of each outcome
expected_value(p, D, g)           expected value of approving, from the PD
breakeven_probability(g)          PD at which approving is worth zero
threshold_sweep(...)              approval rate, default rate among approved,
                                  realised and expected net value per threshold
choose_threshold(sweep)           highest net value with approval rate >= minimum
policy_summary(...)               realised results of one approval policy
classification_metrics(y, p, t)   metrics with "decline" as the positive call
assign_band(p, bands)             risk band names for probabilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)

try:  # imported as part of the src package (notebooks, app)
    from src import config
except ImportError:  # imported from inside the src folder
    import config


def gain_if_repaid(disbursed) -> np.ndarray:
    """Net interest margin earned on a repaid loan (USD)."""
    return config.NET_INTEREST_MARGIN * np.asarray(disbursed, dtype=float)


def loss_if_default(disbursed, portion) -> np.ndarray:
    """Loss on a charged-off loan: share of the unguaranteed principal plus workout cost (USD)."""
    d = np.asarray(disbursed, dtype=float)
    g = np.clip(np.asarray(portion, dtype=float), 0, 1)
    return config.LOSS_GIVEN_DEFAULT * d * (1 - g) + config.WORKOUT_COST_RATE * d


def expected_value(p, disbursed, portion) -> np.ndarray:
    """Expected value of approving a loan with probability of default ``p`` (USD)."""
    p = np.asarray(p, dtype=float)
    return (1 - p) * gain_if_repaid(disbursed) - p * loss_if_default(disbursed, portion)


def breakeven_probability(portion) -> np.ndarray:
    """Probability of default at which approving a loan has zero expected value."""
    g = np.clip(np.asarray(portion, dtype=float), 0, 1)
    nim = config.NET_INTEREST_MARGIN
    return nim / (nim + config.LOSS_GIVEN_DEFAULT * (1 - g) + config.WORKOUT_COST_RATE)


def realised_value(approve, y, disbursed, portion) -> np.ndarray:
    """Value of each loan under a decision, using the actual outcome (0 if declined)."""
    approve = np.asarray(approve, dtype=bool)
    y = np.asarray(y, dtype=int)
    value = np.where(y == 1, -loss_if_default(disbursed, portion), gain_if_repaid(disbursed))
    return np.where(approve, value, 0.0)


def policy_summary(approve, y, disbursed, portion, p=None) -> dict:
    """Realised results of an approval policy on loans with known outcomes."""
    approve = np.asarray(approve, dtype=bool)
    y = np.asarray(y, dtype=int)
    loss = loss_if_default(disbursed, portion)
    out = {
        "approval_rate": float(approve.mean()),
        "default_rate_approved": float(y[approve].mean()) if approve.any() else float("nan"),
        "net_value": float(realised_value(approve, y, disbursed, portion).sum()),
        "chargeoff_loss": float(loss[approve & (y == 1)].sum()),
        "interest_earned": float(gain_if_repaid(disbursed)[approve & (y == 0)].sum()),
        "interest_forgone": float(gain_if_repaid(disbursed)[~approve & (y == 0)].sum()),
    }
    if p is not None:
        out["expected_default_rate_approved"] = float(np.asarray(p)[approve].mean()) if approve.any() else float("nan")
    return out


def threshold_sweep(p, y, disbursed, portion, thresholds=None) -> pd.DataFrame:
    """Results of the rule "approve when p < threshold" for a grid of thresholds."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2) if thresholds is None else np.asarray(thresholds)
    ev = expected_value(p, disbursed, portion)
    value = realised_value(np.ones_like(y, dtype=bool), y, disbursed, portion)
    loss = loss_if_default(disbursed, portion)
    rows = []
    for t in thresholds:
        approve = p < t
        n_app = approve.sum()
        rows.append({
            "threshold": float(t),
            "approval_rate": float(n_app / len(p)),
            "default_rate_approved": float(y[approve].mean()) if n_app else np.nan,
            "expected_default_rate_approved": float(p[approve].mean()) if n_app else np.nan,
            "net_value": float(value[approve].sum()),
            "expected_net_value": float(ev[approve].sum()),
            "chargeoff_loss": float(loss[approve & (y == 1)].sum()),
            "recall_defaults": float(((~approve) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        })
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, min_approval_rate: float = config.MIN_APPROVAL_RATE,
                     value_column: str = "net_value") -> float:
    """Threshold with the highest net value among those meeting the approval-rate rule."""
    ok = sweep[sweep["approval_rate"] >= min_approval_rate]
    if ok.empty:
        raise ValueError("No threshold meets the minimum approval rate")
    return float(ok.loc[ok[value_column].idxmax(), "threshold"])


def classification_metrics(y, p, threshold: float) -> dict:
    """Ranking, calibration and threshold metrics; a prediction of 1 means decline."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def assign_band(p, bands: list[dict]) -> np.ndarray:
    """Risk band name for each probability; ``bands`` is a list of {"name", "min", "max"}."""
    p = np.asarray(p, dtype=float)
    names = np.empty(len(p), dtype=object)
    for band in bands:
        mask = (p >= band["min"]) & (p < band["max"])
        names[mask] = band["name"]
    names[p >= bands[-1]["max"]] = bands[-1]["name"]
    return names
