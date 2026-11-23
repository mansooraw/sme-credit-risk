"""Build the static files the CreditLens app needs.

Run from the repository root:

    python app/build_assets.py raw       # cleaning_params.json and the 50-loan sample files (needs data/raw)
    python app/build_assets.py interim   # interim development model and metadata in app/_dev/ (git-ignored)
    python app/build_assets.py samples   # test_sample_scored.csv and the CSV templates
    python app/build_assets.py aggregates  # aggregate files in app/assets/aggregates/ (needs data/processed)
    python app/build_assets.py all

The aggregate files are small summaries of the full cleaned and engineered
loan table. They let the Market explorer, Bank league and Score a loan pages
work on Streamlit Community Cloud, where data/processed is not available.
Each file is documented in docs/deployment_guide.md and kept under 5 MB.

The interim model has the same interface as the final model described in
section 11 of the brief (Pipeline of src.modelling.build_preprocessor on the
deployment features and LightGBM), so the app can be developed and tested
before notebooks 04 and 05 have produced models/final_model.joblib. The app
always prefers the files in models/ when they exist.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import cleaning, config, features, modelling  # noqa: E402

APP_DIR = REPO_ROOT / "app"
ASSETS = APP_DIR / "assets"
DEV = APP_DIR / "_dev"
TEST_IDS = config.PROCESSED_DATA_DIR / "test_ids.parquet"

DEPLOY = list(getattr(features, "DEPLOYMENT_FEATURES",
                      [c for c in features.FEATURE_COLUMNS if c not in ("ApprovalYear", "ApprovalMonth", "DisbursementLagDays")]))
DEPLOY_NUM = list(getattr(features, "DEPLOYMENT_NUMERIC_FEATURES", [c for c in features.NUMERIC_FEATURES if c in DEPLOY]))
DEPLOY_CAT = list(getattr(features, "DEPLOYMENT_CATEGORICAL_FEATURES", features.CATEGORICAL_FEATURES))
TEXT_COLS = features.CATEGORICAL_FEATURES + [features.BANK_COLUMN]
OUTCOME_COLS = ["MIS_Status", "ChgOffDate", "ChgOffPrinGr", "BalanceGross"]


def test_ids() -> set:
    return set(pd.read_parquet(TEST_IDS)["LoanNr_ChkDgt"])


def load_features(columns=None) -> pd.DataFrame:
    df = pd.read_parquet(config.FEATURES_DATA_PATH, columns=columns)
    for c in TEXT_COLS:
        if c in df:
            df[c] = df[c].astype(str)
    return df


def sample_50_ids() -> list:
    df = load_features(["LoanNr_ChkDgt", "Default"])
    df = df[df["LoanNr_ChkDgt"].isin(test_ids())]
    # 50 test loans, stratified on the outcome so the sample has some defaults
    parts = [g.sample(n=k, random_state=config.RANDOM_STATE)
             for (lab, g), k in zip(df.groupby("Default"), (38, 12))]
    return pd.concat(parts)["LoanNr_ChkDgt"].tolist()


# ---------------------------------------------------------------------------
# Step "raw": cleaning parameters and the raw-layout sample
# ---------------------------------------------------------------------------

def build_raw() -> None:
    """Refit the handle_outliers parameters exactly as clean_raw does.

    Only the columns that the steps before handle_outliers use to remove
    rows, plus the columns handle_outliers needs, are read, to save memory.
    """
    cols = ["LoanNr_ChkDgt", "Name", "City", "State", "ApprovalDate", "GrAppv", "MIS_Status",
            "NoEmp", "NAICS", "Term", "CreateJob", "RetainedJob"]
    raw = cleaning.load_raw(config.RAW_DATA_PATH)
    df = raw[cols].copy()
    df["ApprovalDate"] = cleaning.parse_sba_date(df["ApprovalDate"])
    df["GrAppv"] = cleaning.parse_currency(df["GrAppv"])
    df = cleaning.add_target(df)
    df = cleaning.remove_near_duplicates(df)
    _, params = cleaning.handle_outliers(df)
    params = {k: ({str(a): float(b) for a, b in v.items()} if isinstance(v, dict) else float(v))
              for k, v in params.items()}
    params["_source"] = ("src.cleaning.handle_outliers fitted on the raw data after add_target and "
                         "remove_near_duplicates, as in clean_raw; written by app/build_assets.py")
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "cleaning_params.json").write_text(json.dumps(params, indent=2))
    print("cleaning params:", {k: v for k, v in params.items() if not isinstance(v, dict)})

    ids = sample_50_ids()
    sample = raw[raw["LoanNr_ChkDgt"].isin(ids)].drop(columns=OUTCOME_COLS)
    sample.to_csv(ASSETS / "sample_loans_raw_layout.csv", index=False)
    sample.head(1).to_csv(ASSETS / "template_raw_layout.csv", index=False)
    print("raw sample:", sample.shape)


# ---------------------------------------------------------------------------
# Step "interim": development model and metadata
# ---------------------------------------------------------------------------

def value_parts(disb, portion):
    gain = config.NET_INTEREST_MARGIN * disb
    loss = config.LOSS_GIVEN_DEFAULT * (1 - portion) * disb + config.WORKOUT_COST_RATE * disb
    return gain, loss


def policy(p, y, disb, portion, t):
    gain, loss = value_parts(disb, portion)
    app = p < t
    realised = np.where(y == 1, -loss, gain)
    ev = (1 - p) * gain - p * loss
    return {
        "threshold": float(t),
        "approval_rate": float(app.mean()),
        "expected_default_rate_approved": float(p[app].mean()) if app.any() else 0.0,
        "realised_default_rate_approved": float(y[app].mean()) if app.any() else 0.0,
        "expected_net_value": float(ev[app].sum()),
        "realised_net_value": float(realised[app].sum()),
        "charge_off_loss": float((loss * y * app).sum()),
    }


def build_interim() -> None:
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    from sklearn.pipeline import Pipeline
    import joblib
    import lightgbm
    import shap
    import sklearn

    cols = list(dict.fromkeys(["LoanNr_ChkDgt", "Default", "DisbursementGross"] + DEPLOY))
    df = load_features(cols)
    X_train, X_test, y_train, y_test = modelling.split_data(df, feature_columns=DEPLOY + ["DisbursementGross"],
                                                            save_ids=False, reuse_saved=True)
    disb_test = X_test.pop("DisbursementGross").to_numpy()
    X_train = X_train.drop(columns="DisbursementGross")
    del df

    model = Pipeline([
        ("preprocess", modelling.build_preprocessor(DEPLOY_NUM, DEPLOY_CAT)),
        ("model", LGBMClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=4, verbose=-1)),
    ])
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()
    portion = X_test["Portion"].to_numpy()

    # Comparison model: logistic regression on a 200,000-loan subsample (development only)
    sub = X_train.sample(n=200_000, random_state=config.RANDOM_STATE)
    comp = Pipeline([
        ("preprocess", modelling.build_preprocessor(DEPLOY_NUM, DEPLOY_CAT)),
        ("model", LogisticRegression(max_iter=300, class_weight="balanced")),
    ])
    comp.fit(sub, y_train.loc[sub.index])
    p_comp = comp.predict_proba(X_test)[:, 1]

    # Threshold: maximum realised net value with at least 75% approved
    grid = np.round(np.arange(0.01, 1.0, 0.005), 3)
    rows = [policy(p, y, disb_test, portion, t) for t in grid]
    ok = [r for r in rows if r["approval_rate"] >= config.MIN_APPROVAL_RATE]
    best = max(ok, key=lambda r: r["realised_net_value"])
    t = best["threshold"]
    comp_rows = [policy(p_comp, y, disb_test, portion, tt) for tt in grid]
    comp_ok = [r for r in comp_rows if r["approval_rate"] >= config.MIN_APPROVAL_RATE]
    comp_best = max(comp_ok, key=lambda r: r["realised_net_value"])
    approve_all = policy(p, y, disb_test, portion, 1.01)
    loss_red = 1 - best["charge_off_loss"] / approve_all["charge_off_loss"]

    pred = (p >= t).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "f1": f1_score(y, pred),
        "precision": precision_score(y, pred), "recall": recall_score(y, pred), "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    DEV.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DEV / "final_model.joblib", compress=3)
    joblib.dump(comp, DEV / "comparison_model.joblib", compress=3)

    # SHAP background and global importance on 2,000 stratified training loans
    bg_idx = (pd.concat([X_train, y_train.rename("y")], axis=1).groupby("y", group_keys=False)
              .apply(lambda g: g.sample(frac=2000 / len(X_train), random_state=config.RANDOM_STATE)).index)
    pre = model[:-1]
    Xt = pre.transform(X_train.loc[bg_idx])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = [str(n) for n in pre.get_feature_names_out()]
    pd.DataFrame(Xt, columns=names).to_parquet(DEV / "shap_background.parquet", index=False)
    sv = shap.TreeExplainer(model[-1]).shap_values(Xt)
    sv = sv[-1] if isinstance(sv, list) else sv
    glob = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    glob.sort_values("mean_abs_shap", ascending=False).to_csv(DEV / "shap_global.csv", index=False)

    dummy = "DUMMY interim value"
    meta = {
        "is_interim": True,
        "note": "Interim development artefact written by app/build_assets.py. Not the final model; "
                "comparison, tuning and temporal tables are dummy placeholders.",
        "model_name": "INTERIM development model: LightGBM, 200 trees, default parameters",
        "library_versions": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                             "shap": shap.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "trained_on": date.today().isoformat(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": DEPLOY,
        "categorical_features": DEPLOY_CAT + [features.BANK_COLUMN],
        "numeric_features": DEPLOY_NUM,
        "metrics_test": metrics,
        "metrics_cv": {"roc_auc_mean": 0.0, "roc_auc_std": 0.0, "note": dummy},
        "threshold": t,
        "risk_bands": [
            {"name": "Low", "min": 0.0, "max": 0.10},
            {"name": "Medium", "min": 0.10, "max": 0.25},
            {"name": "High", "min": 0.25, "max": 0.50},
            {"name": "Very high", "min": 0.50, "max": 1.0},
        ],
        "decision_rule": "Approve below the threshold, refer between the threshold and the Very high band, "
                         "decline in the Very high band.",
        "cost_assumptions": {
            "NET_INTEREST_MARGIN": config.NET_INTEREST_MARGIN, "LOSS_GIVEN_DEFAULT": config.LOSS_GIVEN_DEFAULT,
            "WORKOUT_COST_RATE": config.WORKOUT_COST_RATE, "MIN_APPROVAL_RATE": config.MIN_APPROVAL_RATE,
        },
        "business_results": {
            "model_policy": best,
            "approve_all": approve_all,
            "comparison_model": comp_best,
            "loss_reduction_pct": round(100 * loss_red, 2),
        },
        "comparison_table": [
            {"model": "DUMMY row A", "roc_auc": 0.0, "pr_auc": 0.0, "recall": 0.0, "precision": 0.0},
            {"model": "DUMMY row B", "roc_auc": 0.0, "pr_auc": 0.0, "recall": 0.0, "precision": 0.0},
        ],
        "before_after_tuning": [
            {"model": "DUMMY", "stage": "default", "roc_auc": 0.0, "pr_auc": 0.0},
            {"model": "DUMMY", "stage": "tuned", "roc_auc": 0.0, "pr_auc": 0.0},
        ],
        "temporal_validation": {"train_years": "DUMMY", "test_years": "DUMMY", "roc_auc": 0.0, "note": dummy},
        "top_features": glob.sort_values("mean_abs_shap", ascending=False).head(15)
            .rename(columns={"mean_abs_shap": "importance"}).to_dict("records"),
    }
    (DEV / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    print("interim metrics:", metrics, "threshold:", t, "loss reduction %:", round(100 * loss_red, 2))


# ---------------------------------------------------------------------------
# Step "samples": test sample for the deployed threshold page, templates
# ---------------------------------------------------------------------------

def build_samples() -> None:
    import joblib

    cols = list(dict.fromkeys(["LoanNr_ChkDgt", "Default", "DisbursementGross", "GrAppv", "SBA_Appv", "Term",
                               "NAICS", "FranchiseCode"] + DEPLOY))
    df = load_features(cols)
    test = df[df["LoanNr_ChkDgt"].isin(test_ids())].reset_index(drop=True)
    del df
    frac = 20_000 / len(test)
    sample = (test.groupby("Default", group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=config.RANDOM_STATE)))
    sample = sample.head(20_000) if len(sample) > 20_000 else sample
    model_path = config.FINAL_MODEL_PATH if config.FINAL_MODEL_PATH.exists() else DEV / "final_model.joblib"
    model = joblib.load(model_path)
    out = sample[list(dict.fromkeys(["LoanNr_ChkDgt", "Default", "DisbursementGross"] + DEPLOY))].copy()
    out["prob_default_at_build"] = model.predict_proba(out[DEPLOY])[:, 1].round(5)
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(6)
    out.to_csv(ASSETS / "test_sample_scored.csv", index=False)
    print("test sample:", out.shape, "default rate", round(out["Default"].mean(), 4), "scored with", model_path.name)

    # Form-layout sample of the same 50 test loans as the raw-layout sample
    ids = sample_50_ids()
    t50 = test.set_index("LoanNr_ChkDgt").loc[ids].reset_index()
    sector = t50["Sector"]
    form = pd.DataFrame({
        "LoanID": t50["LoanNr_ChkDgt"],
        "GrAppv": t50["GrAppv"], "SBA_Appv": t50["SBA_Appv"], "Portion": "",
        "Term": t50["Term"], "NoEmp": t50["NoEmp"], "CreateJob": t50["CreateJob"], "RetainedJob": t50["RetainedJob"],
        "NewExist": t50["NewExist"], "IsFranchise": np.where(t50["IsFranchise"] == 1, "Y", "N"),
        "UrbanRural": t50["UrbanRural"], "RevLineCr": t50["RevLineCr"], "LowDoc": t50["LowDoc"],
        "Sector": sector, "State": t50["State"], "Bank": t50["Bank"], "BankState": t50["BankState"],
        "DisbursementGross": t50["DisbursementGross"], "Recession": t50["Recession"],
    })
    form.to_csv(ASSETS / "sample_loans_form_layout.csv", index=False)
    template = form.head(1).copy()
    template["LoanID"] = "APP-0001"
    template.to_csv(ASSETS / "template_form_layout.csv", index=False)
    print("form sample:", form.shape)


# ---------------------------------------------------------------------------
# Step "aggregates": summaries of the full data for the deployed app
# ---------------------------------------------------------------------------

AGG_DIR = ASSETS / "aggregates"

# Bands used by the term-by-guarantee heatmap (TermYears and Portion)
TERM_BAND_EDGES = [0, 1, 3, 5, 7, 10, 15, 20, 100]
TERM_BAND_LABELS = ["Under 1 year", "1 to 3 years", "3 to 5 years", "5 to 7 years", "7 to 10 years",
                    "10 to 15 years", "15 to 20 years", "20 years or more"]
PORTION_BAND_EDGES = [0, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9, 1.01]
PORTION_BAND_LABELS = ["Below 50%", "50 to 60%", "60 to 75%", "75 to 80%", "80 to 85%", "85 to 90%", "90% or more"]

# Banks with fewer loans than this are left out of the league table
BANK_MIN_LOANS = 50
# Rows of the reference sample of training loans used for "similar loans"
REFERENCE_ROWS = 50_000


def _write(df: pd.DataFrame, name: str) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    path = AGG_DIR / name
    df.to_parquet(path, index=False, compression="zstd")
    print(f"{name}: {df.shape[0]:,} rows, {path.stat().st_size / 1e6:.2f} MB")


def _counts(g) -> pd.DataFrame:
    return g.agg(loans=("Default", "size"), defaults=("Default", "sum"),
                 disbursed=("DisbursementGross", "sum")).reset_index()


def build_aggregates() -> None:
    cols = ["LoanNr_ChkDgt", "Default", "ApprovalYear", "State", "Sector", "NewExist", "TermYears", "Portion",
            "DisbursementGross", "SBA_Appv", "GrAppv", "Bank", "BankState", "SameStateBank", "Term"] + DEPLOY_NUM
    df = load_features(list(dict.fromkeys(cols)))
    df["Default"] = df["Default"].astype("int32")

    # Market explorer: state x sector x approval year
    sy = _counts(df.groupby(["State", "Sector", "ApprovalYear"], observed=True))
    sy["guaranteed"] = df.groupby(["State", "Sector", "ApprovalYear"], observed=True)["SBA_Appv"].sum().to_numpy()
    _write(sy, "market_state_sector_year.parquet")

    # Market explorer: state x sector x business age (sunburst)
    sa = _counts(df.groupby(["State", "Sector", "NewExist"], observed=True))
    _write(sa, "market_state_sector_age.parquet")

    # Market explorer: state x sector x term band x guarantee band (heatmap)
    df["TermBand"] = pd.cut(df["TermYears"], TERM_BAND_EDGES, labels=TERM_BAND_LABELS, right=False).astype(str)
    df["PortionBand"] = pd.cut(df["Portion"], PORTION_BAND_EDGES, labels=PORTION_BAND_LABELS, right=False).astype(str)
    tp = _counts(df.groupby(["State", "Sector", "TermBand", "PortionBand"], observed=True))
    _write(tp, "market_state_sector_term_portion.parquet")

    # Bank league: one row per lender with at least BANK_MIN_LOANS loans
    g = df.groupby("Bank", observed=True)
    league = pd.DataFrame({
        "loans": g.size(),
        "defaults": g["Default"].sum(),
        "average_loan": g["DisbursementGross"].mean(),
        "total_disbursed": g["DisbursementGross"].sum(),
        "guaranteed_share": g["Portion"].mean(),
        "same_state_share": g["SameStateBank"].mean(),
        "average_term_years": g["TermYears"].mean(),
        "bank_state": g["BankState"].agg(lambda s: s.value_counts().index[0]),
        "states_served": g["State"].nunique(),
        "first_year": g["ApprovalYear"].min(),
        "last_year": g["ApprovalYear"].max(),
    }).reset_index()
    league = league[(league["loans"] >= BANK_MIN_LOANS) & (league["Bank"] != "Unknown")]
    league["default_rate"] = league["defaults"] / league["loans"]
    league = league.sort_values("loans", ascending=False).reset_index(drop=True)
    _write(league, "bank_league.parquet")
    by = _counts(df[df["Bank"].isin(set(league["Bank"]))].groupby(["Bank", "ApprovalYear"], observed=True))
    _write(by, "bank_years.parquet")

    # Reference sample of training loans (not in the test split) for nearest neighbours and percentiles
    train = df[~df["LoanNr_ChkDgt"].isin(test_ids())]
    ref = train.sample(n=min(REFERENCE_ROWS, len(train)), random_state=config.RANDOM_STATE)
    keep = list(dict.fromkeys(DEPLOY_NUM + ["Default", "DisbursementGross", "Term", "Sector", "State", "NewExist"]))
    ref = ref[keep].reset_index(drop=True)
    for c in ref.select_dtypes("float64").columns:
        ref[c] = ref[c].astype("float32")
    for c in ["Sector", "State", "NewExist"]:
        ref[c] = ref[c].astype(str)
    _write(ref, "reference_sample.parquet")

    manifest = {
        "built_on": date.today().isoformat(),
        "source": "data/processed/loans_features.parquet (full cleaned and engineered table)",
        "n_loans": int(len(df)),
        "term_bands": TERM_BAND_LABELS, "portion_bands": PORTION_BAND_LABELS,
        "bank_min_loans": BANK_MIN_LOANS, "reference_rows": int(len(ref)),
        "files": sorted(p.name for p in AGG_DIR.glob("*.parquet")),
    }
    (AGG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("raw", "all"):
        build_raw()
    if step in ("interim", "all"):
        build_interim()
    if step in ("samples", "all"):
        build_samples()
    if step in ("aggregates", "all"):
        build_aggregates()
