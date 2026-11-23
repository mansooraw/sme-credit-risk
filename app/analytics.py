"""Pure computations behind the interactive CreditLens pages.

Nothing here imports Streamlit, so every function can be unit-tested on its
own (tests/test_app_smoke.py) and reused by the pages:

* shareable case links (encode_case, decode_case)
* nearest training loans in the standardised deployment-feature space
* percentile ranks for the applicant radar
* thresholds recommended by business objective
* Monte Carlo simulation of portfolio net value
* stress scenarios and break-even loss given default by segment
* the "Beat the model" game: drawing loans and scoring a round

All money uses the cost matrix of the brief: a repaid loan earns
margin x disbursed, a defaulted loan loses LGD x (1 - guaranteed share) x
disbursed + workout x disbursed, and a declined loan earns and loses nothing.
"""

from __future__ import annotations

from urllib.parse import quote, unquote

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shareable cases
# ---------------------------------------------------------------------------

# Fields of the Score a loan form, in the order used by a case link
CASE_FIELDS = ["GrAppv", "SBA_Appv", "DisbursementGross", "Term", "NoEmp", "CreateJob", "RetainedJob", "NewExist",
               "IsFranchise", "UrbanRural", "RevLineCr", "LowDoc", "Sector", "State", "Bank", "BankState", "Recession"]
CASE_NUMERIC = {"GrAppv": float, "SBA_Appv": float, "DisbursementGross": float, "Term": int, "NoEmp": int,
                "CreateJob": int, "RetainedJob": int, "Recession": int}
CASE_SEPARATOR = "~"


def encode_case(row: dict) -> str:
    """One URL-safe string holding every form field of an application."""
    parts = []
    for f in CASE_FIELDS:
        v = row.get(f, "")
        if f in CASE_NUMERIC:
            v = int(round(float(v))) if v not in ("", None) else ""
        parts.append(quote(str(v), safe=""))
    return CASE_SEPARATOR.join(parts)


def decode_case(text: str | None) -> dict | None:
    """Inverse of encode_case; None when the text is not a valid case."""
    if not text:
        return None
    parts = str(text).split(CASE_SEPARATOR)
    if len(parts) != len(CASE_FIELDS):
        return None
    row = {}
    try:
        for f, raw in zip(CASE_FIELDS, parts):
            value = unquote(raw)
            row[f] = CASE_NUMERIC[f](float(value)) if f in CASE_NUMERIC else value
    except (TypeError, ValueError):
        return None
    if row["GrAppv"] <= 0 or row["DisbursementGross"] <= 0:
        return None
    row["Portion"] = row["SBA_Appv"] / row["GrAppv"]
    row["Recession"] = bool(row["Recession"])
    return row


# ---------------------------------------------------------------------------
# Similar loans and percentiles
# ---------------------------------------------------------------------------

# Heavy-tailed inputs are put on a log scale before standardising, so one
# very large employer does not dominate the distance
LOG_FEATURES = ["NoEmp", "CreateJob", "RetainedJob", "LoanPerEmployee", "JobsPerDollar"]


def transform_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = df[cols].astype(float).to_numpy(copy=True)
    for j, c in enumerate(cols):
        if c in LOG_FEATURES:
            out[:, j] = np.log1p(np.clip(out[:, j], 0, None))
    return out


def prepare_reference(ref: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised reference matrix plus the mean and scale used."""
    X = transform_numeric(ref, cols)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return ((X - mean) / scale).astype(np.float32), mean, scale


def nearest(Z: np.ndarray, mean: np.ndarray, scale: np.ndarray, row: pd.DataFrame, cols: list[str],
            k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Indices and Euclidean distances of the k reference rows closest to ``row``."""
    z = ((transform_numeric(row, cols)[0] - mean) / scale).astype(np.float32)
    d = np.sqrt(((Z - z) ** 2).sum(axis=1))
    idx = np.argpartition(d, k)[:k]
    idx = idx[np.argsort(d[idx])]
    return idx, d[idx]


def percentile_rank(values: np.ndarray, x: float) -> float:
    """Share of values below x plus half the ties, in percent (0 to 100)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return 50.0
    return float(100 * ((v < x).sum() + 0.5 * (v == x).sum()) / len(v))


# ---------------------------------------------------------------------------
# Cost matrix and policies
# ---------------------------------------------------------------------------

def gain_loss(disbursed, portion, margin: float, lgd: float, workout: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(disbursed, dtype=float)
    g = np.clip(np.asarray(portion, dtype=float), 0, 1)
    return margin * d, lgd * (1 - g) * d + workout * d


def policy_curve(p, y, gain, loss, grid) -> pd.DataFrame:
    """Approval rate, realised net value and recall for each threshold in grid (vectorised)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(p)
    ps = p[order]
    realised = np.where(y == 1, -loss, gain)[order]
    cum_value = np.concatenate([[0.0], np.cumsum(realised)])
    cum_def = np.concatenate([[0], np.cumsum(y[order])])
    n, total_def = len(p), int(y.sum())
    k = np.searchsorted(ps, np.asarray(grid, dtype=float), side="left")  # approved: p < t
    missed = cum_def[k]
    return pd.DataFrame({
        "threshold": np.asarray(grid, dtype=float),
        "approval_rate": k / n if n else 0.0,
        "realised_net_value": cum_value[k],
        "recall": (total_def - missed) / total_def if total_def else 0.0,
        "default_rate_approved": np.where(k > 0, missed / np.maximum(k, 1), 0.0),
    })


def objective_thresholds(p, y, gain, loss, min_approval: float, target_recall: float = 0.90,
                         grid=None) -> dict:
    """Thresholds that meet three business objectives.

    max_value      the threshold with the highest realised net value, no constraint
    approval_floor the highest net value with at least ``min_approval`` approved
    recall_target  the highest threshold (most approvals) that still stops at
                   least ``target_recall`` of the defaults
    """
    grid = np.round(np.arange(0.01, 1.0, 0.005), 3) if grid is None else np.asarray(grid)
    c = policy_curve(p, y, gain, loss, grid)
    out = {"max_value": float(c.loc[c["realised_net_value"].idxmax(), "threshold"])}
    ok = c[c["approval_rate"] >= min_approval]
    out["approval_floor"] = float(ok.loc[ok["realised_net_value"].idxmax(), "threshold"]) if len(ok) else out["max_value"]
    rec = c[c["recall"] >= target_recall]
    out["recall_target"] = float(rec["threshold"].max()) if len(rec) else float(c["threshold"].min())
    return out


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo(p, gain, loss, thresholds, runs: int = 1000, seed: int = 42, chunk: int = 100) -> np.ndarray:
    """Simulated portfolio net value, shape (len(thresholds), runs).

    In each run every loan defaults independently with probability equal to
    its PD (a Bernoulli draw). A loan approved at threshold t (PD below t)
    then adds its gain if it is repaid and subtracts its loss if it defaults.
    The same random draws are used for every threshold (common random
    numbers), so the differences between thresholds are not noise.
    """
    p = np.asarray(p, dtype=np.float32)
    gain = np.asarray(gain, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    approved = (p[:, None] < thresholds[None, :]).astype(np.float64)      # n x T
    base = gain @ approved                                                 # value if nothing defaults
    hit = (gain + loss)[:, None] * approved                                # n x T, lost when a loan defaults
    rng = np.random.default_rng(seed)
    out = np.empty((len(thresholds), runs))
    done = 0
    while done < runs:
        m = min(chunk, runs - done)
        defaults = (rng.random((m, len(p)), dtype=np.float32) < p).astype(np.float64)
        out[:, done:done + m] = (base[None, :] - defaults @ hit).T
        done += m
    return out


def distribution_summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    return {"p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "mean": float(v.mean()), "std": float(v.std()), "prob_loss": float((v < 0).mean())}


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------

def expected_net(p, gain, loss) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return (1 - p) * gain - p * loss


def break_even_lgd(p, disbursed, portion, margin: float, workout: float) -> float:
    """Loss given default at which the expected net value of a group of loans is zero.

    Solves sum((1-p) m d) - sum(p (L (1-g) d + w d)) = 0 for L. Returns inf
    when the group has no unguaranteed exposure at risk.
    """
    p = np.asarray(p, dtype=float)
    d = np.asarray(disbursed, dtype=float)
    g = np.clip(np.asarray(portion, dtype=float), 0, 1)
    exposure = float((p * (1 - g) * d).sum())
    if exposure <= 0:
        return float("inf")
    return float(((1 - p) * margin * d).sum() - (p * workout * d).sum()) / exposure


# ---------------------------------------------------------------------------
# Beat the model
# ---------------------------------------------------------------------------

GAME_ROUND = 5
GAME_DEFAULTS = 2  # each round holds this many loans that were charged off


def draw_game_loans(df: pd.DataFrame, seed: int, n: int = GAME_ROUND, n_defaults: int = GAME_DEFAULTS) -> pd.DataFrame:
    """Draw n held-out loans, n_defaults of which defaulted, in random order."""
    rng = np.random.default_rng(seed)
    bad = df.index[df["Default"] == 1].to_numpy()
    good = df.index[df["Default"] == 0].to_numpy()
    n_defaults = min(n_defaults, len(bad), n)
    pick = np.concatenate([rng.choice(bad, n_defaults, replace=False), rng.choice(good, n - n_defaults, replace=False)])
    rng.shuffle(pick)
    return df.loc[pick]


def game_outcome(approve: bool, default: int, gain: float, loss: float) -> float:
    """Realised value of one decision: gain or loss if approved, nothing if declined."""
    if not approve:
        return 0.0
    return -float(loss) if int(default) == 1 else float(gain)


def game_summary(user_approves, model_approves, defaults, gains, losses) -> dict:
    """Accuracy and money for the player and the model over one round.

    A decision is correct when an approved loan was repaid or a declined
    loan defaulted.
    """
    u = np.asarray(user_approves, dtype=bool)
    m = np.asarray(model_approves, dtype=bool)
    y = np.asarray(defaults, dtype=int)
    g = np.asarray(gains, dtype=float)
    lo = np.asarray(losses, dtype=float)
    value = lambda a: float(np.where(a, np.where(y == 1, -lo, g), 0.0).sum())  # noqa: E731
    correct = lambda a: int(((a & (y == 0)) | (~a & (y == 1))).sum())  # noqa: E731
    best = float(np.where(y == 0, g, 0.0).sum())
    return {
        "n": int(len(y)),
        "user_correct": correct(u), "model_correct": correct(m),
        "user_accuracy": correct(u) / len(y) if len(y) else 0.0,
        "model_accuracy": correct(m) / len(y) if len(y) else 0.0,
        "user_value": value(u), "model_value": value(m), "best_value": best,
        "approve_all_value": value(np.ones_like(u)),
    }
