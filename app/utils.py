"""Shared helpers for the CreditLens Streamlit app (CRISP-DM Deployment).

Everything the pages need lives here so that each page stays short:

* artefact loading (final model in models/ with a fallback to the interim
  development model in app/_dev/), cached with st.cache_resource / st.cache_data
* turning form inputs or uploaded CSV files into the cleaned loan table with
  the functions in src/cleaning.py and the stored cleaning parameters
* feature building with src/features.py, scoring, risk bands and decisions
* the cost matrix from src/config.py: expected value and expected loss
* local explanations (SHAP) and global importance mapped to readable labels
* portfolio summaries at a given threshold

No number shown in the app is typed in: each comes from the metadata, the
configuration or a computation in this module.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import streamlit as st  # noqa: E402

from src import cleaning, config, features  # noqa: E402

ASSETS_DIR = APP_DIR / "assets"
DEV_DIR = APP_DIR / "_dev"
LOGO_PATH = ASSETS_DIR / "creditlens_logo.svg"
ICON_PATH = ASSETS_DIR / "creditlens_icon.svg"
CLEANING_PARAMS_PATH = ASSETS_DIR / "cleaning_params.json"
TEST_SAMPLE_PATH = ASSETS_DIR / "test_sample_scored.csv"
SAMPLE_RAW_PATH = ASSETS_DIR / "sample_loans_raw_layout.csv"
SAMPLE_FORM_PATH = ASSETS_DIR / "sample_loans_form_layout.csv"
TEMPLATE_RAW_PATH = ASSETS_DIR / "template_raw_layout.csv"
TEMPLATE_FORM_PATH = ASSETS_DIR / "template_form_layout.csv"
MODEL_CARD_PATH = config.DOCS_DIR / "model_card.md"
TEST_IDS_PATH = config.PROCESSED_DATA_DIR / "test_ids.parquet"

REPOSITORY_URL = "https://github.com/mansooraw/sme-credit-risk"

ARTEFACT_FILES = {
    "model": "final_model.joblib",
    "comparison": "comparison_model.joblib",
    "metadata": "model_metadata.json",
    "shap_background": "shap_background.parquet",
    "shap_global": "shap_global.csv",
}

NEW_BANK_OPTION = "Other / new bank"
NEW_BANK_NAME = "OTHER OR NEW BANK"  # not a training bank: encoded with BankSize 0 and the overall rate

# Calendar features that the deployed model does not use (fallback when the
# constant is missing from src/features.py)
CALENDAR_FEATURES = ["ApprovalYear", "ApprovalMonth", "DisbursementLagDays"]

# Loan start dates used for form inputs. The deployed model has no calendar
# features, so the date only drives the Recession indicator: a loan starting
# in mid-2008 overlaps the recession window, one starting in 2015 cannot.
RECESSION_START_DATE = pd.Timestamp("2008-06-01")
NO_RECESSION_START_DATE = pd.Timestamp("2015-01-01")

# Representative NAICS code for each sector name (first code of the sector)
SECTOR_TO_NAICS = {}
for _code, _name in config.NAICS_SECTORS.items():
    SECTOR_TO_NAICS.setdefault(_name, int(_code) * 10_000)
SECTOR_TO_NAICS[config.UNKNOWN_SECTOR] = 0

# App palette: one accent, three semantic colours, two neutrals. The keys are
# kept from the notebooks' palette so that pages and charts share one vocabulary.
COLOURS = {
    "navy": "#1F4E8C",   # accent, primary series
    "teal": "#1A7F5A",   # good: approve, repaid
    "amber": "#B7791F",  # caution: refer
    "red": "#C0392B",    # bad: decline, default
    "grey": "#8B94A3",
    "slate": "#4F5B6B",
    "dark_red": "#8E2A1F",
}
DECISION_COLOURS = {"Approve": COLOURS["teal"], "Refer": COLOURS["amber"], "Decline": COLOURS["red"]}
BAND_COLOURS = [COLOURS["teal"], COLOURS["amber"], COLOURS["red"], COLOURS["dark_red"]]

# Columns of the two accepted CSV layouts
RAW_REQUIRED = [
    "State", "Bank", "BankState", "NAICS", "ApprovalDate", "Term", "NoEmp", "NewExist",
    "CreateJob", "RetainedJob", "FranchiseCode", "UrbanRural", "RevLineCr", "LowDoc",
    "DisbursementGross", "GrAppv", "SBA_Appv",
]
RAW_OPTIONAL = ["LoanNr_ChkDgt", "Name", "City", "Zip", "ApprovalFY", "DisbursementDate",
                "ChgOffDate", "BalanceGross", "ChgOffPrinGr", "MIS_Status"]
FORM_REQUIRED = [
    "GrAppv", "Term", "NoEmp", "CreateJob", "RetainedJob", "NewExist", "IsFranchise", "UrbanRural",
    "RevLineCr", "LowDoc", "Sector", "State", "Bank", "BankState", "DisbursementGross", "Recession",
]
FORM_OPTIONAL = ["LoanID", "SBA_Appv", "Portion"]  # one of SBA_Appv or Portion is required

FORM_COLUMN_HELP = {
    "LoanID": "Optional identifier, copied to the output.",
    "GrAppv": "Gross amount approved in USD.",
    "SBA_Appv": "Guaranteed amount in USD (or give Portion instead).",
    "Portion": "Guaranteed share between 0 and 1 (used when SBA_Appv is empty).",
    "Term": "Term in months, 1 to 480.",
    "NoEmp": "Number of employees.",
    "CreateJob": "Jobs to be created.",
    "RetainedJob": "Jobs retained.",
    "NewExist": "Existing or New.",
    "IsFranchise": "Y or N.",
    "UrbanRural": "Urban, Rural or Unknown.",
    "RevLineCr": "Revolving line of credit: Y or N.",
    "LowDoc": "LowDoc programme: Y or N.",
    "Sector": "Industry sector name as listed in the app.",
    "State": "Two-letter borrower state.",
    "Bank": "Lending bank name; any name not seen in training is treated as a new bank.",
    "BankState": "Two-letter bank state.",
    "DisbursementGross": "Amount disbursed in USD.",
    "Recession": "1 to stress-test the loan as if it ran through the 2007 to 2009 recession, else 0.",
}


# ---------------------------------------------------------------------------
# Artefact location and loading
# ---------------------------------------------------------------------------

def artefact_source() -> dict:
    """Where the model artefacts are read from.

    The final artefacts in models/ are used when both the model and the
    metadata exist; otherwise the interim development artefacts in app/_dev/.
    Swapping in the final model therefore needs no code change.
    """
    final_ok = (config.MODELS_DIR / ARTEFACT_FILES["model"]).exists() and (
        config.MODELS_DIR / ARTEFACT_FILES["metadata"]).exists()
    base = config.MODELS_DIR if final_ok else DEV_DIR
    return {"dir": base, "is_dev": not final_ok}


def artefact_path(key: str) -> Path | None:
    """Path of one artefact in the active folder, or None if it is missing."""
    path = artefact_source()["dir"] / ARTEFACT_FILES[key]
    return path if path.exists() else None


def _mtime(path: Path | None) -> float:
    return path.stat().st_mtime if path is not None and path.exists() else 0.0


def model_version_key() -> str:
    """Changes whenever a different model file is loaded (used as a cache key)."""
    path = artefact_path("model")
    return f"{path}:{_mtime(path)}"


@st.cache_resource(show_spinner="Loading the model")
def _load_joblib(path_str: str, mtime: float):
    return joblib.load(path_str)


def load_model():
    path = artefact_path("model")
    if path is None:
        raise FileNotFoundError(
            "No model found. Expected models/final_model.joblib or app/_dev/final_model.joblib "
            "(run `python app/build_assets.py interim`).")
    return _load_joblib(str(path), _mtime(path))


def load_comparison_model():
    path = artefact_path("comparison")
    return None if path is None else _load_joblib(str(path), _mtime(path))


@st.cache_resource(show_spinner=False)
def _load_json(path_str: str, mtime: float) -> dict:
    with open(path_str, encoding="utf-8") as fh:
        return json.load(fh)


def load_metadata() -> dict:
    path = artefact_path("metadata")
    return {} if path is None else _load_json(str(path), _mtime(path))


def load_schema() -> dict:
    return _load_json(str(config.FEATURE_SCHEMA_PATH), _mtime(config.FEATURE_SCHEMA_PATH))


def load_cleaning_params() -> dict:
    """Outlier caps and imputation medians fitted in src.cleaning.handle_outliers.

    Taken from the metadata when the final model stores them, otherwise from
    app/assets/cleaning_params.json (written by app/build_assets.py from the
    raw data with the same cleaning steps as notebook 02).
    """
    meta = load_metadata()
    for key in ("cleaning_params", "cleaning_parameters"):
        if isinstance(meta.get(key), dict) and meta[key]:
            return meta[key]
    return _load_json(str(CLEANING_PARAMS_PATH), _mtime(CLEANING_PARAMS_PATH))


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_shap_global() -> pd.DataFrame | None:
    """Global importance as a two-column table (feature, importance), or None."""
    path = artefact_path("shap_global")
    if path is None:
        return None
    df = _read_csv(str(path), _mtime(path))
    text_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not text_cols or not num_cols:
        return None
    feature_col = next((c for c in text_cols if "feature" in c.lower()), text_cols[0])
    value_col = next((c for c in num_cols if "shap" in c.lower() or "importance" in c.lower()), num_cols[0])
    return df[[feature_col, value_col]].rename(columns={feature_col: "feature", value_col: "importance"})


def schema_fields() -> dict:
    """Feature and raw form field definitions from feature_schema.json, by name."""
    schema = load_schema()
    fields = {f["name"]: f for f in schema.get("features", [])}
    raw = {f["name"]: f for f in schema.get("raw_form_fields", [])}
    return {"features": fields, "raw": raw}


def deployment_features() -> list[str]:
    """Columns the deployed model receives."""
    meta = load_metadata()
    if isinstance(meta.get("features"), list) and meta["features"]:
        return list(meta["features"])
    if hasattr(features, "DEPLOYMENT_FEATURES"):
        return list(features.DEPLOYMENT_FEATURES)
    return [c for c in features.FEATURE_COLUMNS if c not in CALENDAR_FEATURES]


# ---------------------------------------------------------------------------
# Thresholds, bands and decisions
# ---------------------------------------------------------------------------

DEFAULT_BANDS = [
    {"name": "Low", "min": 0.0, "max": 0.10},
    {"name": "Medium", "min": 0.10, "max": 0.25},
    {"name": "High", "min": 0.25, "max": 0.50},
    {"name": "Very high", "min": 0.50, "max": 1.0},
]


def risk_bands() -> list[dict]:
    bands = load_metadata().get("risk_bands") or DEFAULT_BANDS
    bands = sorted(bands, key=lambda b: float(b.get("min", 0) or 0))
    out = []
    for i, b in enumerate(bands):
        lo = float(b.get("min", 0) or 0)
        hi = b.get("max")
        hi = 1.0 if hi is None else float(hi)
        out.append({"name": str(b.get("name", f"Band {i + 1}")), "min": lo, "max": hi,
                    "colour": BAND_COLOURS[min(i, len(BAND_COLOURS) - 1)]})
    return out


def threshold() -> float:
    meta = load_metadata()
    return float(meta.get("threshold", 0.5))


def decline_threshold() -> float:
    """Probability at or above which the recommendation is Decline.

    Uses ``decline_threshold`` from the metadata when present; otherwise the
    lower edge of the highest risk band. Between the approval threshold and
    this value the loan is referred for manual review.
    """
    meta = load_metadata()
    if meta.get("decline_threshold") is not None:
        return max(float(meta["decline_threshold"]), threshold())
    return max(risk_bands()[-1]["min"], threshold())


def band_for(p: float) -> dict:
    bands = risk_bands()
    for b in bands:
        if p < b["max"] or b is bands[-1]:
            if p >= b["min"]:
                return b
    return bands[0]


def band_names(p: np.ndarray) -> np.ndarray:
    bands = risk_bands()
    edges = [b["min"] for b in bands[1:]]
    idx = np.searchsorted(edges, p, side="right")
    return np.array([bands[i]["name"] for i in idx])


def decisions(p: np.ndarray, t: float | None = None, decline_at: float | None = None) -> np.ndarray:
    t = threshold() if t is None else t
    d = decline_threshold() if decline_at is None else decline_at
    p = np.asarray(p, dtype=float)
    return np.where(p < t, "Approve", np.where(p < d, "Refer", "Decline"))


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------

def cost_assumptions() -> dict:
    """Cost constants: from the metadata when stored there, else src/config.py."""
    base = {
        "NET_INTEREST_MARGIN": config.NET_INTEREST_MARGIN,
        "LOSS_GIVEN_DEFAULT": config.LOSS_GIVEN_DEFAULT,
        "WORKOUT_COST_RATE": config.WORKOUT_COST_RATE,
        "MIN_APPROVAL_RATE": config.MIN_APPROVAL_RATE,
    }
    stored = load_metadata().get("cost_assumptions") or {}
    for key in list(base):
        for k, v in stored.items():
            if k.upper() == key and isinstance(v, (int, float)):
                base[key] = float(v)
    return base


def value_components(disbursed, portion) -> tuple[np.ndarray, np.ndarray]:
    """Gain if repaid and loss if charged off, per loan, in USD."""
    c = cost_assumptions()
    disbursed = np.asarray(disbursed, dtype=float)
    portion = np.clip(np.asarray(portion, dtype=float), 0, 1)
    gain = c["NET_INTEREST_MARGIN"] * disbursed
    loss = c["LOSS_GIVEN_DEFAULT"] * (1 - portion) * disbursed + c["WORKOUT_COST_RATE"] * disbursed
    return gain, loss


def expected_value(p, disbursed, portion) -> tuple[np.ndarray, np.ndarray]:
    """Expected net value of approving and expected loss, per loan, in USD."""
    gain, loss = value_components(disbursed, portion)
    p = np.asarray(p, dtype=float)
    return (1 - p) * gain - p * loss, p * loss


def cost_table() -> pd.DataFrame:
    c = cost_assumptions()
    return pd.DataFrame([
        {"Assumption": "Net interest margin on a repaid loan", "Value": f"{c['NET_INTEREST_MARGIN']:.0%} of the disbursed amount"},
        {"Assumption": "Loss given default", "Value": f"{c['LOSS_GIVEN_DEFAULT']:.0%} of the unguaranteed principal"},
        {"Assumption": "Workout and collection cost on default", "Value": f"{c['WORKOUT_COST_RATE']:.0%} of the disbursed amount"},
        {"Assumption": "Declined loan", "Value": "No gain and no loss"},
        {"Assumption": "Minimum approval rate when setting the threshold", "Value": f"{c['MIN_APPROVAL_RATE']:.0%}"},
    ])


def formula_text() -> str:
    c = cost_assumptions()
    return (
        f"Gain if repaid  = {c['NET_INTEREST_MARGIN']:.2f} x disbursed\n"
        f"Loss if default = {c['LOSS_GIVEN_DEFAULT']:.2f} x (1 - guaranteed share) x disbursed"
        f" + {c['WORKOUT_COST_RATE']:.2f} x disbursed\n"
        "Expected loss   = PD x loss if default\n"
        "Expected value  = (1 - PD) x gain if repaid - PD x loss if default"
    )


# ---------------------------------------------------------------------------
# Inputs to the cleaned loan table
# ---------------------------------------------------------------------------

def _yes_no(value) -> str:
    text = str(value).strip().upper()
    if text in {"Y", "YES", "1", "1.0", "TRUE"}:
        return "Y"
    if text in {"N", "NO", "0", "0.0", "FALSE"}:
        return "N"
    return "Unknown"


def form_to_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn rows in the form layout into the cleaned-table columns.

    The form asks for business terms (sector name, franchise yes or no,
    recession stress) rather than source codes, so each is translated to the
    code that src/cleaning.py and src/features.py expect.
    """
    out = pd.DataFrame(index=df.index)
    gr = pd.to_numeric(df["GrAppv"], errors="coerce").astype(float)
    sba = pd.to_numeric(df["SBA_Appv"], errors="coerce") if "SBA_Appv" in df else pd.Series(np.nan, index=df.index)
    if "Portion" in df:
        sba = sba.fillna(pd.to_numeric(df["Portion"], errors="coerce") * gr)
    out["GrAppv"] = gr
    out["SBA_Appv"] = sba.astype(float)
    out["DisbursementGross"] = pd.to_numeric(df["DisbursementGross"], errors="coerce").astype(float)
    for col in ["Term", "NoEmp", "CreateJob", "RetainedJob"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    new_exist = df["NewExist"].astype(str).str.strip().str.capitalize()
    out["NewExist"] = new_exist.where(new_exist.isin(["Existing", "New"]), "Unknown")
    out["FranchiseCode"] = np.where(df["IsFranchise"].map(_yes_no) == "Y", 10_000, 1)
    urban = df["UrbanRural"].astype(str).str.strip().str.capitalize()
    out["UrbanRural"] = urban.where(urban.isin(["Urban", "Rural"]), "Unknown")
    rev = df["RevLineCr"].astype(str).str.strip().str.upper()
    out["RevLineCr"] = rev.where(rev == "T", rev.map(_yes_no))  # T is kept as its own level
    out["LowDoc"] = df["LowDoc"].map(_yes_no)
    sector = df["Sector"].astype(str).str.strip()
    out["NAICS"] = sector.map(SECTOR_TO_NAICS).fillna(0).astype(int)
    out["State"] = df["State"].astype(str).str.strip().str.upper()
    out["BankState"] = df["BankState"].astype(str).str.strip().str.upper()
    bank = df["Bank"].astype(str).str.strip()
    out["Bank"] = bank.replace({NEW_BANK_OPTION: NEW_BANK_NAME, "": NEW_BANK_NAME, "nan": NEW_BANK_NAME})
    stress = df["Recession"].map(_yes_no) == "Y"
    start = pd.Series(np.where(stress, RECESSION_START_DATE, NO_RECESSION_START_DATE), index=df.index)
    out["ApprovalDate"] = pd.to_datetime(start)
    out["DisbursementDate"] = pd.to_datetime(start)
    return out


def _parse_dates(s: pd.Series) -> pd.Series:
    """Dates in the source format (28-Feb-97) or any ISO-like format."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    parsed = cleaning.parse_sba_date(s.astype("string"))
    retry = parsed.isna() & s.notna()
    if retry.any():
        parsed[retry] = pd.to_datetime(s[retry], errors="coerce")
    return parsed


def raw_to_clean(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Clean rows in the raw SBA layout with the src.cleaning steps.

    The same steps as notebook 02 are applied (type parsing, code cleaning,
    logical checks, outlier treatment with the stored parameters, missing
    text values), except the two row filters: a new application has no
    outcome yet and must not be dropped as a near-duplicate.
    """
    raw = df.copy()
    for col in RAW_OPTIONAL:
        if col not in raw:
            raw[col] = np.nan
    raw["ApprovalDate"] = _parse_dates(raw["ApprovalDate"])
    raw["DisbursementDate"] = _parse_dates(raw["DisbursementDate"])
    raw["ChgOffDate"] = _parse_dates(raw["ChgOffDate"])
    if raw["ApprovalFY"].isna().all():
        raw["ApprovalFY"] = raw["ApprovalDate"].dt.year.astype("Int64").astype(str)
    raw["MIS_Status"] = raw["MIS_Status"].fillna("Unknown")
    parsed = cleaning.parse_types(raw)
    coded = cleaning.clean_codes(parsed)
    checked = cleaning.apply_logical_checks(coded)
    # Dates after the end of the historical data are valid for new applications
    for col in ["ApprovalDate", "DisbursementDate"]:
        late = checked[col].isna() & parsed[col].notna() & (parsed[col] > cleaning.VALID_DATE_MAX)
        checked.loc[late, col] = parsed.loc[late, col]
    treated, _ = cleaning.handle_outliers(checked, params=params)
    return cleaning.fill_missing_categoricals(treated)


def detect_layout(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    if set(RAW_REQUIRED) <= cols:
        return "raw"
    if set(FORM_REQUIRED) <= cols and ({"SBA_Appv", "Portion"} & cols):
        return "form"
    return None


def missing_columns(df: pd.DataFrame) -> dict:
    cols = set(df.columns)
    form_missing = [c for c in FORM_REQUIRED if c not in cols]
    if not ({"SBA_Appv", "Portion"} & cols):
        form_missing.append("SBA_Appv or Portion")
    return {"raw": [c for c in RAW_REQUIRED if c not in cols], "form": form_missing}


def validate_clean(clean: pd.DataFrame) -> pd.Series:
    """Reason a cleaned row cannot be scored, or an empty string."""
    reasons = pd.Series("", index=clean.index, dtype=object)

    def flag(mask, text):
        mask = pd.Series(mask, index=clean.index).fillna(True)
        reasons[mask & (reasons == "")] = text

    flag(~(clean["GrAppv"] > 0), "approved amount missing or not positive")
    flag(~(clean["DisbursementGross"] > 0), "disbursed amount missing or not positive")
    flag(~(clean["SBA_Appv"] >= 0), "guaranteed amount missing or negative")
    flag(clean["SBA_Appv"] > clean["GrAppv"] * 1.0000001, "guaranteed amount above approved amount")
    flag(~((clean["Term"] >= 1) & (clean["Term"] <= 480)), "term outside 1 to 480 months")
    for col in ["NoEmp", "CreateJob", "RetainedJob"]:
        flag(~(clean[col] >= 0), f"{col} missing or negative")
    flag(clean["ApprovalDate"].isna(), "approval date missing or unreadable")
    return reasons


def build_model_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Engineered features (src.features.build_features) for the model."""
    feats = features.build_features(clean)
    return feats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def predict(model, feats: pd.DataFrame) -> np.ndarray:
    cols = deployment_features()
    return model.predict_proba(feats[cols])[:, 1]


def score_table(feats: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    ev, el = expected_value(p, feats["DisbursementGross"], feats["Portion"])
    return pd.DataFrame({
        "probability_of_default": p,
        "risk_band": band_names(p),
        "decision": decisions(p),
        "expected_net_value_usd": ev,
        "expected_loss_usd": el,
    }, index=feats.index)


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

SHORT_LABELS = {
    "BankDefaultRate": "Bank default rate", "BankSize": "Bank size", "SameStateBank": "Same-state lender",
    "Recession": "Recession overlap", "DisbursedToApprovedRatio": "Disbursed / approved", "JobsPerDollar": "Jobs per USD 100k",
    "LoanPerEmployee": "Loan per employee", "LogDisbursement": "Disbursed", "RevLineCr": "Revolving line",
    "DisbursementDateMissing": "No disbursement date", "RealEstate": "Real-estate term", "NewBusiness": "New business",
    "IsFranchise": "Franchise", "Portion": "Guarantee", "TermYears": "Term", "Sector": "Sector", "State": "State",
    "BankState": "Bank state", "UrbanRural": "Location", "NewExist": "Business age", "LowDoc": "LowDoc",
    "NoEmp": "Employees", "CreateJob": "Jobs created", "RetainedJob": "Jobs retained", "Bank": "Bank",
}


def short_label(name: str) -> str:
    return SHORT_LABELS.get(name, feature_label(name))


def feature_label(name: str) -> str:
    info = features.FEATURE_INFO.get(name)
    if name == "LogDisbursement":
        return "Disbursed amount"
    return info[0] if info else name


def split_pipeline(model):
    """Preprocessing part and final estimator of a fitted Pipeline."""
    if hasattr(model, "steps") and len(model.steps) > 1:
        return model[:-1], model.steps[-1][1]
    return None, model


def find_bank_encoder(obj):
    """The fitted BankEncoder inside a pipeline, searched recursively."""
    if isinstance(obj, features.BankEncoder):
        return obj
    children = []
    if hasattr(obj, "steps"):
        children = [s[1] for s in obj.steps]
    elif hasattr(obj, "transformers_"):
        children = [t[1] for t in obj.transformers_]
    for child in children:
        found = find_bank_encoder(child)
        if found is not None:
            return found
    return None


def output_feature_names(model) -> list[str]:
    pre, est = split_pipeline(model)
    if pre is not None:
        try:
            return [str(n) for n in pre.get_feature_names_out()]
        except Exception:  # noqa: BLE001
            pass
    n = getattr(est, "n_features_in_", 0)
    return [f"feature_{i}" for i in range(n)]


def source_feature(output_name: str) -> str:
    """Model input column that a preprocessed column came from."""
    name = output_name.split("__")[-1]
    inputs = deployment_features() + list(features.BANK_FEATURES)
    if name in inputs:
        return name
    cats = sorted(features.CATEGORICAL_FEATURES, key=len, reverse=True)
    for c in cats:
        if name.startswith(c + "_"):
            return c
    return name


@st.cache_resource(show_spinner=False)
def _explainer(model_key: str):
    import shap
    _, est = split_pipeline(load_model())
    return shap.TreeExplainer(est)


def warm_explainer():
    """Create (and cache) the SHAP explainer for the loaded model."""
    return _explainer(model_version_key())


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def local_contributions(model, feats_row: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per-feature contributions (log-odds) for one loan.

    SHAP values from shap.TreeExplainer on the final estimator and the
    preprocessed row; one-hot columns are summed back to their input
    feature. Falls back to LightGBM's built-in contributions, then raises.
    """
    pre, est = split_pipeline(model)
    cols = deployment_features()
    X = feats_row[cols]
    Xt = _dense(pre.transform(X)) if pre is not None else X.to_numpy()
    method = "SHAP (TreeExplainer)"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = _explainer(model_version_key()).shap_values(Xt)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        row = values[0]
    except Exception:  # noqa: BLE001
        if not hasattr(est, "booster_"):
            raise
        row = est.booster_.predict(Xt, pred_contrib=True)[0][:-1]
        method = "LightGBM contributions"
    names = output_feature_names(model)
    frame = pd.DataFrame({"output": names[: len(row)], "contribution": row[: len(names)]})
    frame["feature"] = frame["output"].map(source_feature)
    agg = frame.groupby("feature", sort=False)["contribution"].sum().reset_index()
    agg["label"] = agg["feature"].map(feature_label)
    agg["value"] = [display_value(f, feats_row.iloc[0], model) for f in agg["feature"]]
    agg = agg.reindex(agg["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return agg, method


def display_value(feature: str, row: pd.Series, model=None) -> str:
    """Readable value of one model input for one loan."""
    try:
        if feature in ("BankSize", "BankDefaultRate"):
            enc = find_bank_encoder(model) if model is not None else None
            if enc is None:
                return ""
            size, rate = enc.transform(pd.DataFrame({features.BANK_COLUMN: [row[features.BANK_COLUMN]]}))[0]
            return f"{size:,.0f} loans" if feature == "BankSize" else f"{rate:.1%}"
        if feature == "LogDisbursement":
            return f"USD {np.expm1(row[feature]):,.0f}"
        if feature == "Portion":
            return f"{row[feature]:.0%}"
        if feature == "TermYears":
            return f"{row[feature]:.1f} years"
        if feature == "LoanPerEmployee":
            return f"USD {row[feature]:,.0f}"
        if feature in ("JobsPerDollar", "DisbursedToApprovedRatio"):
            return f"{row[feature]:.2f}"
        if feature in features.ENGINEERED_BINARY:
            return "Yes" if int(row[feature]) == 1 else "No"
        if feature in ("RevLineCr", "LowDoc"):
            return {"Y": "Yes", "N": "No", "T": "code T"}.get(str(row[feature]), str(row[feature]))
        if feature == features.BANK_COLUMN:
            return "Other or new bank" if row[feature] == NEW_BANK_NAME else str(row[feature]).title()
        value = row[feature]
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value:,.0f}"
        return str(value)
    except Exception:  # noqa: BLE001
        return ""


def explanation_sentences(contrib: pd.DataFrame, n: int = 3) -> list[str]:
    """Plain sentences on the main drivers of one score."""
    sentences = []
    top = contrib.head(n)
    scale = contrib["contribution"].abs().sum() or 1.0
    for _, r in top.iterrows():
        share = abs(r["contribution"]) / scale
        strength = "strongly " if share >= 0.25 else ("" if share >= 0.10 else "slightly ")
        direction = "raises" if r["contribution"] > 0 else "lowers"
        label = str(r["label"]).split(" (")[0]
        value = f" ({r['value']})" if r["value"] else ""
        sentences.append(f"{label}{value} {strength}{direction} the estimated risk of default.")
    up = contrib.loc[contrib["contribution"] > 0, "contribution"].sum()
    down = -contrib.loc[contrib["contribution"] < 0, "contribution"].sum()
    if up > down:
        sentences.append("Taken together, the factors that raise the risk outweigh those that lower it.")
    else:
        sentences.append("Taken together, the factors that lower the risk outweigh those that raise it.")
    return sentences


def global_importance(top: int = 15) -> pd.DataFrame | None:
    """Global mean absolute SHAP, aggregated to input features with labels."""
    g = load_shap_global()
    if g is None:
        return None
    g = g.copy()
    g["source"] = g["feature"].astype(str).map(source_feature)
    agg = g.groupby("source", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    agg["label"] = agg["source"].map(feature_label)
    return agg.head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Held-out portfolio for the threshold page
# ---------------------------------------------------------------------------

def _portfolio_columns() -> list[str]:
    return list(dict.fromkeys(deployment_features() + ["Default", "DisbursementGross", "Portion", "State", "Sector"]))


@st.cache_data(show_spinner="Scoring the held-out test portfolio")
def scored_portfolio(model_key: str) -> tuple[pd.DataFrame, str]:
    """Test-set loans with their predicted probability of default.

    Uses the full held-out test split from data/processed when it is
    available (local runs), otherwise the committed 20,000-loan sample of the
    same split in app/assets/test_sample_scored.csv (deployed app). The
    probabilities are always recomputed with the loaded model.
    """
    cols = _portfolio_columns()
    if config.FEATURES_DATA_PATH.exists() and TEST_IDS_PATH.exists():
        df = pd.read_parquet(config.FEATURES_DATA_PATH, columns=list(dict.fromkeys(cols + ["LoanNr_ChkDgt"])))
        ids = set(pd.read_parquet(TEST_IDS_PATH)["LoanNr_ChkDgt"])
        df = df[df["LoanNr_ChkDgt"].isin(ids)].reset_index(drop=True)
        source = f"full held-out test split ({len(df):,} loans, data/processed)"
    else:
        df = pd.read_csv(TEST_SAMPLE_PATH)
        source = f"stratified sample of the held-out test split ({len(df):,} loans, app/assets)"
    for c in features.CATEGORICAL_FEATURES + [features.BANK_COLUMN]:
        if c in df:
            df[c] = df[c].astype(str)
    df["p"] = load_model().predict_proba(df[deployment_features()])[:, 1]
    return df[["p", "Default", "DisbursementGross", "Portion", "State", "Sector"]].copy(), source


def policy_summary(p, y, disbursed, portion, t: float) -> dict:
    """Portfolio outcome of approving every loan with PD below ``t``."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    gain, loss = value_components(disbursed, portion)
    approved = p < t
    realised = np.where(y == 1, -loss, gain)
    expected, _ = expected_value(p, disbursed, portion)
    n_app = int(approved.sum())
    loss_all = float((loss * y).sum())
    loss_policy = float((loss * y * approved).sum())
    tp = int(((~approved) & (y == 1)).sum())
    fp = int(((~approved) & (y == 0)).sum())
    fn = int((approved & (y == 1)).sum())
    tn = int((approved & (y == 0)).sum())
    return {
        "threshold": t,
        "n": len(p),
        "approval_rate": n_app / len(p) if len(p) else 0.0,
        "expected_default_rate_approved": float(p[approved].mean()) if n_app else 0.0,
        "realised_default_rate_approved": float(y[approved].mean()) if n_app else 0.0,
        "realised_net_value": float(realised[approved].sum()),
        "expected_net_value": float(expected[approved].sum()),
        "net_value_approve_all": float(realised.sum()),
        "loss_approve_all": loss_all,
        "loss_policy": loss_policy,
        "loss_avoided": loss_all - loss_policy,
        "loss_reduction": (loss_all - loss_policy) / loss_all if loss_all else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }


def threshold_curve(p, y, disbursed, portion, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        s = policy_summary(p, y, disbursed, portion, float(t))
        rows.append({"threshold": t, "approval_rate": s["approval_rate"], "realised_net_value": s["realised_net_value"],
                     "loss_reduction": s["loss_reduction"]})
    return pd.DataFrame(rows)


def best_threshold(p, y, disbursed, portion, min_approval: float | None = None) -> float:
    """Threshold that maximises realised net value at the minimum approval rate."""
    min_approval = cost_assumptions()["MIN_APPROVAL_RATE"] if min_approval is None else min_approval
    curve = threshold_curve(p, y, disbursed, portion, grid=np.round(np.arange(0.01, 1.0, 0.005), 3))
    ok = curve[curve["approval_rate"] >= min_approval]
    ok = ok if len(ok) else curve
    return float(ok.loc[ok["realised_net_value"].idxmax(), "threshold"])


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def find_value(obj, names: list[str]):
    """First value under any of ``names`` (case-insensitive) in nested dicts."""
    wanted = {n.lower() for n in names}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in wanted and not isinstance(v, (dict, list)):
                return v
        for v in obj.values():
            found = find_value(v, names)
            if found is not None:
                return found
    return None


def model_policy_results(meta: dict) -> dict:
    """The business results of the model policy from metadata['business_results']."""
    br = meta.get("business_results") or {}
    if not isinstance(br, dict):
        return {}
    for k, v in br.items():
        key = str(k).lower()
        if isinstance(v, dict) and any(w in key for w in ("model", "final", "lgbm", "lightgbm", "policy")) and \
                not any(w in key for w in ("approve_all", "approve all", "comparison", "logistic", "baseline")):
            return v
    return br


def flatten(d, prefix: str = "") -> list[tuple[str, object]]:
    rows = []
    if isinstance(d, dict):
        for k, v in d.items():
            name = f"{prefix}{k}"
            if isinstance(v, dict):
                rows.extend(flatten(v, name + " / "))
            elif isinstance(v, list) and v and not isinstance(v[0], dict):
                rows.append((name, ", ".join(str(x) for x in v)))
            elif not isinstance(v, list):
                rows.append((name, v))
    return rows


ACRONYMS = {"roc auc": "ROC-AUC", "pr auc": "PR-AUC", "f1": "F1", "cv": "CV", "pct": "(%)", "usd": "(USD)", "auc": "AUC"}


def prettify_key(key: str) -> str:
    text = str(key).replace("_", " ").strip()
    low = text.lower()
    for k, v in ACRONYMS.items():
        if low == k or low.startswith(k + " ") or low.endswith(" " + k) or f" {k} " in low:
            low = low.replace(k, v)
    return low[:1].upper() + low[1:]


def pretty_table(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Readable column names and rounded numbers for a metadata table."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(decimals)
    out.columns = [prettify_key(c) for c in out.columns]
    return out


def format_business_value(column: str, value):
    """Format one business-results value by what its name says it holds."""
    name = str(column).lower()
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    if "rate" in name or "share" in name:
        return f"{value:.1%}" if abs(value) <= 1.5 else f"{value:.1f}%"
    if any(w in name for w in ("value", "loss", "usd", "gain", "profit")):
        return fmt_usd(float(value))
    if "threshold" in name:
        return f"{value:.3f}"
    return f"{value:,.4g}"


def records_table(rows) -> pd.DataFrame | None:
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return pd.DataFrame(rows)
    if isinstance(rows, dict) and rows:
        try:
            return pd.DataFrame(rows)
        except ValueError:
            return pd.DataFrame(flatten(rows), columns=["Item", "Value"])
    return None


def fmt_usd(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9:
        return f"{sign}USD {x / 1e9:,.2f} bn"
    if x >= 1e6:
        return f"{sign}USD {x / 1e6:,.1f} m"
    return f"{sign}USD {x:,.0f}"


def fmt_metric(v, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pct:
        return f"{v:.1%}" if abs(v) <= 1.5 else f"{v:.1f}%"
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# Page furniture: theme, icons, headings, tiles, chips, charts
# ---------------------------------------------------------------------------

THEME_PATH = ASSETS_DIR / "theme.css"
LOGO_LIGHT_PATH = ASSETS_DIR / "creditlens_logo_light.svg"
FONT_STACK = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
MONO_STACK = "ui-monospace, SF Mono, Menlo, Consolas, monospace"
# Light and dark colour sets. INK, MUTED, BORDER, GRID and CARD are read
# through the module __getattr__ below, so pages that use U.INK follow the
# dark-mode toggle without any change.
LIGHT = {"INK": "#141A22", "MUTED": "#5C6675", "BORDER": "#E2E5EA", "GRID": "#EEF0F3", "CARD": "#FFFFFF",
         "BG": "#F4F5F7", "HOVER": "#141A22", "HOVER_TEXT": "#FFFFFF"}
DARK = {"INK": "#E6EAF0", "MUTED": "#9AA5B5", "BORDER": "#27303C", "GRID": "#222A35", "CARD": "#161C24",
        "BG": "#0F1419", "HOVER": "#E6EAF0", "HOVER_TEXT": "#141A22"}


def is_dark() -> bool:
    try:
        return bool(st.session_state.get("dark_mode", False))
    except Exception:  # noqa: BLE001
        return False


def pal() -> dict:
    """Current colour set (light or dark)."""
    return DARK if is_dark() else LIGHT


def __getattr__(name: str):
    if name in LIGHT:
        return pal()[name]
    raise AttributeError(name)



# Simple 24 x 24 stroke icons drawn for the app (no external icon set)
_ICON_PATHS = {
    "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h4"/>',
    "gauge": '<path d="M4 16a8 8 0 1 1 16 0"/><path d="M12 16l4-5"/><circle cx="12" cy="16" r="1.4"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.8"/>',
    "bank": '<path d="M3 10l9-6 9 6"/><path d="M5 10v8M9.5 10v8M14.5 10v8M19 10v8"/><path d="M3 21h18"/>',
    "briefcase": '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/><path d="M3 13h18"/>',
    "coins": '<ellipse cx="9" cy="7" rx="6" ry="3"/><path d="M3 7v5c0 1.7 2.7 3 6 3s6-1.3 6-3V7"/><path d="M9 15v2c0 1.7 2.7 3 6 3s6-1.3 6-3v-5c0-1.6-2.4-2.9-5.5-3"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "map": '<path d="M9 4L3 6v14l6-2 6 2 6-2V4l-6 2z"/><path d="M9 4v14M15 6v14"/>',
    "grid": '<rect x="3" y="3" width="8" height="8" rx="2"/><rect x="13" y="3" width="8" height="8" rx="2"/><rect x="3" y="13" width="8" height="8" rx="2"/><rect x="13" y="13" width="8" height="8" rx="2"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "shield": '<path d="M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    "users": '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.8-3.5 3.4-5.5 6.5-5.5s5.7 2 6.5 5.5"/><circle cx="17" cy="9" r="2.5"/><path d="M17 14.5c2.3.3 3.9 2 4.5 4.5"/>',
    "book": '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5"/><path d="M8 7h7"/>',
    "phone": '<rect x="6" y="2" width="12" height="20" rx="3"/><path d="M11 18h2"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "scale": '<path d="M12 3v18M5 21h14"/><path d="M5 7h14"/><path d="M5 7l-3 7a3 3 0 0 0 6 0zM19 7l-3 7a3 3 0 0 0 6 0z"/>',
    "spark": '<path d="M3 17l5-5 4 4 8-9"/><path d="M15 7h5v5"/>',
    "download": '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
}


def icon(name: str, size: int = 20, colour: str = "#2A9D8F", stroke: float = 1.9) -> str:
    """Inline SVG icon."""
    body = _ICON_PATHS.get(name, _ICON_PATHS["info"])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{colour}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{body}</svg>')


TOKENS_PATH = ASSETS_DIR / "tokens.css"


@st.cache_data(show_spinner=False)
def _theme_css(mtime: float) -> str:
    tokens = TOKENS_PATH.read_text(encoding="utf-8") if TOKENS_PATH.exists() else ""
    return tokens + "\n" + THEME_PATH.read_text(encoding="utf-8")


def inject_theme() -> None:
    """Inject app/assets/tokens.css and theme.css (read once and cached)."""
    st.markdown(f"<style>{_theme_css(_mtime(THEME_PATH) + _mtime(TOKENS_PATH))}</style>", unsafe_allow_html=True)


def apply_style() -> None:
    """Kept for compatibility: the theme is injected by the entry script."""
    return None


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def model_meta_line() -> str:
    """Short right-hand line for page headers: model, training date, test size."""
    meta = load_metadata()
    name = str(meta.get("model_name", "Model"))
    short = "LightGBM" if "lightgbm" in name.lower() else name.split("(")[0].strip()
    bits = [f"<b>{short}</b>"]
    if meta.get("trained_on"):
        bits.append(f"trained {meta['trained_on']}")
    if isinstance(meta.get("n_test"), int):
        bits.append(f"{meta['n_test']:,} held-out loans")
    bits.append(f"approve below {threshold():.1%}")
    return " &middot; ".join(bits)


def page_header(title: str, intro: str | None = None, eyebrow: str | None = None, meta: str | bool = True) -> None:
    """Title row: title and one-line intro on the left, model line on the right.

    ``meta`` is the right-hand text (default: the model line); False hides it.
    ``eyebrow`` is accepted for compatibility and not shown.
    """
    left = f'<div><h1 class="cl-page-title">{title}</h1>'
    if intro:
        left += f'<div class="cl-page-intro">{intro}</div>'
    if artefact_source()["is_dev"]:
        left += '<div class="cl-devnote">Interim development model: figures are for testing only</div>'
    left += "</div>"
    right = ""
    if meta:
        right = f'<div class="cl-page-meta">{model_meta_line() if meta is True else meta}</div>'
    html(f'<div class="cl-page-head">{left}{right}</div>')


def section_title(text: str, icon_name: str | None = None) -> None:
    """Small uppercase section label with a rule. ``icon_name`` is accepted and ignored."""
    html(f'<div class="cl-section-title"><span>{text}</span></div>')


def page_link(page: str, label: str, icon: str | None = None) -> None:
    """st.page_link that degrades to bold text when a page runs on its own (tests)."""
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:  # noqa: BLE001
        st.markdown(f"**{label}**")


def switch_page(page: str) -> None:
    """st.switch_page that is safe when a page runs on its own (tests)."""
    try:
        st.switch_page(page)
    except Exception:  # noqa: BLE001
        st.toast("Use the sidebar to open this page.")


def hex_rgba(colour: str, alpha: float) -> str:
    c = colour.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def chip(text: str, colour: str, solid: bool = False) -> str:
    """Flat label: a coloured dot and text inside a hairline border. ``solid`` is accepted and ignored."""
    return f'<span class="cl-chip"><span class="sw" style="background:{colour}"></span>{text}</span>'


def pill(text: str, colour: str, dark_text: bool = False) -> str:
    return chip(text, colour, solid=True)


def decision_chip(decision: str) -> str:
    """Flat decision tag: coloured dot, uppercase text, border in the decision colour."""
    colour = DECISION_COLOURS[decision]
    return (f'<span class="cl-decision" style="color:{colour};border-color:{hex_rgba(colour, 0.5)}">'
            f'<i style="background:{colour}"></i>{decision}</span>')


def how_tip(text: str) -> str:
    """Small "i" control that explains how a number is computed.

    A <details> element, so it opens on tap and with the keyboard without any
    JavaScript; CSS also opens it on hover with a mouse.
    """
    import html as _html
    return (f'<details class="cl-how"><summary aria-label="How this is computed">i</summary>'
            f'<div class="cl-how-body">{_html.escape(text)}</div></details>')


def tiles(items: list[dict], cols: int = 4, min_width: int | None = None, compact: bool = False,
          glass: bool = False) -> None:
    """KPI tiles. Each item: label, value, optional sub, accent, help and count (for the count-up).

    ``count`` is a dict {"to", "decimals", "prefix", "suffix"}; the final value
    is always written into the HTML, so the tile is correct without JavaScript.
    ``help`` becomes a tap-to-open note on how the number is computed.
    """
    cells = []
    for it in items:
        accent = it.get("accent", COLOURS["teal"])
        value = it["value"]
        count = it.get("count")
        if count:
            attrs = (f' class="cl-tile-value cl-countup" data-to="{count["to"]}" data-dec="{count.get("decimals", 0)}"'
                     f' data-pre="{count.get("prefix", "")}" data-suf="{count.get("suffix", "")}"')
        else:
            attrs = ' class="cl-tile-value"'
        sub = f'<div class="cl-tile-sub">{it["sub"]}</div>' if it.get("sub") else ""
        tip = how_tip(it["help"]) if it.get("help") else ""
        cells.append(f'<div class="cl-tile">{tip}'
                     f'<div class="cl-tile-label">{it["label"]}</div>'
                     f'<div{attrs}>{value}</div>{sub}</div>')
    minimum = min_width or {1: 100, 2: 150, 3: 150, 4: 150, 5: 140}.get(cols, 140)
    cls = "cl-tiles" + (" compact" if compact else "")
    html(f'<div class="{cls}" style="grid-template-columns:repeat(auto-fit,minmax({minimum}px,1fr))">{"".join(cells)}</div>')


def table_html(columns: list[str], rows: list[list], numeric: set[int] | None = None, caption: str | None = None) -> str:
    """Compact HTML table (class cl-table). ``numeric`` holds the right-aligned column indexes."""
    numeric = numeric or set()
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{c}</th>' for i, c in enumerate(columns))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{v}</td>' for i, v in enumerate(r)) + "</tr>"
                   for r in rows)
    cap = f'<div class="cl-tile-label" style="margin-bottom:6px">{caption}</div>' if caption else ""
    return (f'{cap}<div class="cl-table-wrap"><table class="cl-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def skeleton(height: int = 320, label: str = "Computing") -> str:
    """Shimmering placeholder shown while a heavy block computes."""
    return (f'<div class="cl-skeleton" style="height:{height}px" role="status" aria-live="polite">'
            f'<div class="cl-skel-bar w60"></div><div class="cl-skel-bar w90"></div><div class="cl-skel-bar w75"></div>'
            f'<div class="cl-skel-label">{label}</div></div>')


# Page runtime: marks the document as script-enabled, reveals cards and tiles
# as they scroll into view (IntersectionObserver adds a class, CSS does the
# transition) and counts KPI values up when they first appear. Installed once
# on the parent document; a MutationObserver picks up elements added by
# later reruns. Without JavaScript nothing is hidden and every value is final.
_RUNTIME_JS = """
<script>
(function () {
  var w = window.parent, doc = w && w.document;
  if (!doc) return;
  if (w.__clRuntime) { w.__clRuntime.scan(); return; }
  function expandNav() {
    var b = doc.querySelector('[data-testid="stSidebarNavViewButton"]');
    if (b && /more/i.test(b.innerText) && !b.__clDone) { b.__clDone = true; b.click(); }
  }
  w.__clRuntime = {scan: expandNav};
  expandNav();
  new w.MutationObserver(expandNav).observe(doc.body, {childList: true, subtree: true});
})();
</script>
"""


def runtime() -> None:
    """Install the page runtime (zero-height component, hidden by CSS)."""
    try:
        import streamlit.components.v1 as components
        components.html(_RUNTIME_JS, height=0)
    except Exception:  # noqa: BLE001
        pass


def countup() -> None:
    """Kept for the existing pages: the runtime counts tiles up as they appear."""
    runtime()


def sidebar_status() -> None:
    meta = load_metadata()
    dev = artefact_source()["is_dev"]
    name = meta.get("model_name", "Model not available")
    html(
        f'<div class="cl-status"><div class="cl-status-head"><span class="cl-dot{" dev" if dev else ""}"></span>'
        f'{"Interim model" if dev else "Model in service"}</div>'
        f'<div class="cl-status-name">{name}</div>'
        f'<div class="cl-status-row"><span>Trained</span><b>{meta.get("trained_on", "n/a")}</b></div>'
        f'<div class="cl-status-row"><span>Approve below</span><b>{threshold():.1%}</b></div>'
        f'<div class="cl-status-row"><span>Decline from</span><b>{decline_threshold():.1%}</b></div></div>'
    )


# Plotly template shared by every chart
def _axis() -> dict:
    return dict(showgrid=True, gridcolor=pal()["GRID"], zeroline=False, linecolor=pal()["BORDER"], ticks="", automargin=True,
                tickfont=dict(color=pal()["MUTED"], size=11, family=FONT_STACK),
                title=dict(font=dict(color=pal()["MUTED"], size=11, family=FONT_STACK), standoff=8))


def plotly_template():
    import plotly.graph_objects as go
    return go.layout.Template(layout=dict(
        font=dict(family=FONT_STACK, size=12, color=pal()["INK"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[COLOURS["navy"], COLOURS["teal"], COLOURS["amber"], COLOURS["red"], COLOURS["grey"], COLOURS["slate"]],
        xaxis=_axis(), yaxis=_axis(),
        margin=dict(l=10, r=10, t=36, b=30),
        title=dict(font=dict(size=13, color=pal()["INK"], family=FONT_STACK), x=0.01, xanchor="left", y=0.97),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11, color=pal()["MUTED"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=pal()["HOVER"], bordercolor=pal()["HOVER"], font=dict(color=pal()["HOVER_TEXT"], family=FONT_STACK, size=11), align="left"),
        bargap=0.32,
    ))


def chart(fig, height: int | None = None, key: str | None = None, on_select=None, selection_mode="points"):
    """Render a Plotly figure with the CreditLens template at full width.

    With ``on_select="rerun"`` the chart is clickable and the selection event
    is returned (``event.selection.points``), which the pages use to drill down.
    """
    tpl = plotly_template()
    fig.update_layout(template=tpl)
    # Explicit values as well as the template: the front end fills unset layout keys with its own theme
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT_STACK, color=pal()["INK"], size=12),
                      hoverlabel=dict(bgcolor=pal()["HOVER"], bordercolor=pal()["HOVER"], font=dict(color=pal()["HOVER_TEXT"], family=FONT_STACK, size=12), align="left"))
    for ax in ("xaxis", "yaxis"):
        current = getattr(fig.layout, ax)
        if current is not None and current.visible is not False:
            fig.update_layout({ax: dict(gridcolor=current.gridcolor or pal()["GRID"], linecolor=pal()["BORDER"], zerolinecolor=pal()["GRID"],
                                        tickfont=dict(color=pal()["MUTED"], size=11, family=FONT_STACK))})
    if fig.layout.title.text:
        top = fig.layout.margin.t if fig.layout.margin.t is not None else 36
        fig.update_layout(title=dict(font=dict(size=13, color=pal()["INK"], family=FONT_STACK), x=0.01, xanchor="left",
                                     yref="container", y=0.98, yanchor="top"),
                          margin=dict(t=max(top, 64)))
        if fig.layout.legend.y is None:
            fig.update_layout(legend=dict(y=1.02, yanchor="bottom", x=0, xanchor="left"))
    if height:
        fig.update_layout(height=height)
    if is_dark():
        _darken(fig)
    kwargs = {}
    if on_select is not None:
        kwargs = {"on_select": on_select, "selection_mode": selection_mode}
    return st.plotly_chart(fig, use_container_width=True, theme=None, key=key,
                           config={"displayModeBar": False, "responsive": True}, **kwargs)


_LIGHT_ONLY = {"#FFFFFF", "#FFF", "WHITE", "#EEF0F3", "#F4F5F7", "#EEF2F7", "#F7F9FC"}


def _darken(fig) -> None:
    """Swap the light-only colours that pages write into traces for dark-mode ones."""
    card, ink = DARK["CARD"], DARK["INK"]

    def swap(c):
        return card if isinstance(c, str) and c.upper() in _LIGHT_ONLY else c

    for tr in fig.data:
        marker = getattr(tr, "marker", None)
        if marker is not None and getattr(marker, "line", None) is not None:
            col = marker.line.color
            if isinstance(col, str):
                marker.line.color = swap(col)
        if getattr(tr, "fillcolor", None) and str(tr.fillcolor).upper() in _LIGHT_ONLY:
            tr.fillcolor = DARK["GRID"]
        line = getattr(tr, "line", None)
        if line is not None and isinstance(getattr(line, "color", None), str) and line.color.upper() in _LIGHT_ONLY:
            line.color = card
        navy_dark = "#6FA3E6"
        if line is not None and isinstance(getattr(line, "color", None), str) and line.color.upper() == COLOURS["navy"].upper():
            line.color = navy_dark
        if marker is not None:
            mc = getattr(marker, "color", None)
            if isinstance(mc, str) and mc.upper() == COLOURS["navy"].upper():
                marker.color = navy_dark
            elif isinstance(mc, (list, tuple)):
                marker.color = [navy_dark if isinstance(c, str) and c.upper() == COLOURS["navy"].upper() else c for c in mc]
            cols = getattr(marker, "colors", None)
            if cols is not None and not isinstance(cols, str):
                try:
                    marker.colors = [swap(c) for c in cols]
                except TypeError:
                    pass
        tf = getattr(tr, "textfont", None)
        if tf is not None and isinstance(tf.color, str) and tf.color.upper() == LIGHT["INK"].upper():
            tf.color = ink
    for ann in fig.layout.annotations or []:
        if ann.font is not None and isinstance(ann.font.color, str) and ann.font.color.upper() == LIGHT["INK"].upper():
            ann.font.color = ink
    if fig.layout.geo is not None:
        fig.update_geos(bgcolor="rgba(0,0,0,0)", lakecolor=card, landcolor=DARK["GRID"], subunitcolor=DARK["BORDER"])


def gauge_figure(p: float, t: float, compact: bool = False):
    """Semi-circular gauge: band arcs in the risk colours, needle at PD, tick at the threshold."""
    import plotly.graph_objects as go
    fig = go.Figure()

    def arc(a0, a1, r0, r1, n=40):
        th = np.linspace(np.pi * (1 - a0), np.pi * (1 - a1), n)
        xs = np.concatenate([r1 * np.cos(th), r0 * np.cos(th[::-1])])
        ys = np.concatenate([r1 * np.sin(th), r0 * np.sin(th[::-1])])
        return np.append(xs, xs[0]), np.append(ys, ys[0])

    bx, by = arc(0, 1, 0.66, 1.0)
    fig.add_trace(go.Scatter(x=bx, y=by, fill="toself", fillcolor="#EEF0F3", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    for b in risk_bands():
        x, y = arc(b["min"], min(b["max"], 1.0), 0.70, 0.96)
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor=b["colour"], line=dict(width=1.5, color="#FFFFFF"),
                                 hoverinfo="text", text=f"{b['name']}: {b['min']:.0%} to {b['max']:.0%}", showlegend=False))
    ang = np.pi * (1 - p)
    fig.add_trace(go.Scatter(x=[-0.05 * np.cos(ang), 0.86 * np.cos(ang)], y=[-0.05 * np.sin(ang), 0.86 * np.sin(ang)],
                             mode="lines", line=dict(color=pal()["INK"], width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=12, color=pal()["INK"], line=dict(color="#FFFFFF", width=2)),
                             hoverinfo="skip", showlegend=False))
    ta = np.pi * (1 - t)
    fig.add_trace(go.Scatter(x=[0.62 * np.cos(ta), 1.04 * np.cos(ta)], y=[0.62 * np.sin(ta), 1.04 * np.sin(ta)],
                             mode="lines", line=dict(color=pal()["INK"], width=2.5, dash="dot"), hoverinfo="text",
                             text=f"Approval threshold {t:.1%}", showlegend=False))
    for v, label in ([] if compact else [(0, "0%"), (0.5, "50%"), (1, "100%")]):
        a = np.pi * (1 - v)
        fig.add_annotation(x=1.13 * np.cos(a), y=1.10 * np.sin(a) - (0.06 if v in (0, 1) else 0), text=label,
                           showarrow=False, font=dict(size=11, color=pal()["MUTED"], family=MONO_STACK))
    fig.add_annotation(x=0.0, y=0.36, text="" if compact else f"threshold {t:.1%}", showarrow=False,
                       font=dict(size=11, color=pal()["MUTED"], family=MONO_STACK))
    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.25, 1.25], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.2, 1.22], scaleanchor="x", scaleratio=1, fixedrange=True),
        margin=dict(l=0, r=0, t=6, b=0), height=230, showlegend=False,
    )
    return fig


def _driver_svg(contrib, n: int = 8) -> str:
    """Inline SVG bar chart of the main score drivers (for the printable memo)."""
    import html as _html
    if contrib is None or not len(contrib):
        return ""
    top = contrib.head(n)
    vmax = float(top["contribution"].abs().max()) or 1.0
    row_h, width, mid = 26, 680, 340
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {row_h * len(top) + 10}" width="100%" '
             'role="img" aria-label="Main drivers of the score">',
             f'<line x1="{mid}" y1="0" x2="{mid}" y2="{row_h * len(top) + 10}" stroke="#8D99AE" stroke-width="1"/>']
    for i, (_, r) in enumerate(top.iterrows()):
        v = float(r["contribution"])
        w = abs(v) / vmax * (mid - 170)
        y = 6 + i * row_h
        colour = COLOURS["red"] if v > 0 else COLOURS["teal"]
        x = mid if v > 0 else mid - w
        label = _html.escape(f"{short_label(r['feature'])}" + (f": {r['value']}" if r["value"] else ""))[:40]
        tx, anchor = (mid - 8, "end") if v > 0 else (mid + 8, "start")
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w, 1):.1f}" height="16" rx="5" fill="{colour}"/>')
        parts.append(f'<text x="{tx}" y="{y + 12}" font-size="11" fill="#0B1F3A" text-anchor="{anchor}">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def memo_html(inputs: dict, p: float, band: dict, decision: str, ev: float, el: float, gain: float, loss: float,
              drivers: list[str], t: float, d: float, contrib=None, printable: bool = False) -> str:
    """One-page credit memo as a self-contained HTML file.

    ``printable`` adds A4 print rules, a print button and a chart of the main
    drivers, so the browser's Print dialog saves a clean PDF.
    """
    import datetime as _dt
    import html as _html

    meta = load_metadata()
    logo = LOGO_PATH.read_text(encoding="utf-8") if LOGO_PATH.exists() else "CreditLens"
    rows = "".join(f"<tr><th>{_html.escape(str(k))}</th><td>{_html.escape(str(v))}</td></tr>" for k, v in inputs.items())
    driver_items = "".join(f"<li>{_html.escape(s)}</li>" for s in drivers)
    colour = DECISION_COLOURS[decision]
    fg = LIGHT["INK"] if decision == "Refer" else "#FFFFFF"
    today = _dt.date.today().strftime("%d %B %Y")
    model = f"{meta.get('model_name', 'n/a')} (trained {meta.get('trained_on', 'n/a')})"
    dev_note = ("<p class='note'>Scored with the interim development model; not for credit decisions.</p>"
                if artefact_source()["is_dev"] else "")
    chart_svg = _driver_svg(contrib) if printable else ""
    print_css = ("@page{{size:A4;margin:16mm 14mm}} .bar{{position:sticky;top:0;background:#0B1F3A;color:#fff;padding:10px 16px;"
                 "display:flex;justify-content:space-between;align-items:center;font-size:13px}} "
                 ".bar button{{background:#2A9D8F;color:#fff;border:0;border-radius:999px;padding:8px 18px;font-weight:700;"
                 "cursor:pointer;font-family:inherit}} @media print{{.bar{{display:none}} .page{{max-width:none;padding:0}} "
                 "h2{{break-after:avoid}} table,ul,svg{{break-inside:avoid}} .k{{-webkit-print-color-adjust:exact;"
                 "print-color-adjust:exact}} .dec{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}"
                 ).replace("{{", "{").replace("}}", "}") if printable else ""
    print_bar = ('<div class="bar"><span>CreditLens credit memo, print-ready</span>'
                 '<button onclick="window.print()">Print or save as PDF</button></div>') if printable else ""
    return f"""<!doctype html><html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CreditLens credit memo</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
body{{font-family:'Plus Jakarta Sans',Arial,sans-serif;color:#0B1F3A;margin:0;background:#F7F9FC}}
.page{{max-width:780px;margin:24px auto;background:#fff;border:1px solid #E3E8EF;border-radius:14px;padding:36px 40px}}
header{{display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid #0B1F3A;padding-bottom:14px}}
header svg{{height:40px;width:auto}} .meta{{text-align:right;font-size:12px;color:#5B6B82}}
h1{{font-size:22px;margin:22px 0 4px}} h2{{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:#5B6B82;margin:24px 0 8px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .k{{border:1px solid #E3E8EF;border-radius:10px;padding:10px 12px}}
.k .l{{font-size:11px;color:#5B6B82;text-transform:uppercase;letter-spacing:.05em}} .k .v{{font-family:'JetBrains Mono',monospace;font-size:18px;margin-top:4px}}
.dec{{display:inline-block;padding:6px 16px;border-radius:999px;background:{colour};color:{fg};font-weight:800}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #EEF2F7}} th{{color:#5B6B82;font-weight:600;width:45%}}
li{{margin-bottom:4px;font-size:13px}} .f{{font-family:'JetBrains Mono',monospace;font-size:12px;background:#F7F9FC;padding:10px;border-radius:8px;white-space:pre-wrap}}
.note{{font-size:12px;color:#7A5A12;background:#FBF3DC;padding:8px 10px;border-radius:8px}} footer{{margin-top:24px;font-size:11px;color:#8D99AE}}
@media print{{body{{background:#fff}} .page{{border:none;margin:0}}}}
{print_css}
</style></head><body>{print_bar}<div class="page">
<header>{logo}<div class="meta">Credit memo<br>{today}</div></header>
<h1>Recommendation: <span class="dec">{decision}</span></h1>
<p style="color:#5B6B82;font-size:13px">Approve below {t:.1%}, refer from {t:.1%} to {d:.1%}, decline from {d:.1%}.</p>{dev_note}
<div class="grid">
<div class="k"><div class="l">Probability of default</div><div class="v">{p:.1%}</div></div>
<div class="k"><div class="l">Risk band</div><div class="v">{band['name']}</div></div>
<div class="k"><div class="l">Expected net value</div><div class="v">{fmt_usd(ev)}</div></div>
<div class="k"><div class="l">Expected loss</div><div class="v">{fmt_usd(el)}</div></div></div>
<h2>Main drivers</h2>{chart_svg}<ul>{driver_items}</ul>
<h2>Application</h2><table>{rows}</table>
<h2>Expected value</h2><div class="f">{_html.escape(formula_text())}
Gain if repaid {fmt_usd(gain)}; loss if charged off {fmt_usd(loss)}.</div>
<footer>Model: {_html.escape(model)}. CreditLens is a decision-support tool; the credit officer makes the final decision.</footer>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Example applications and form scoring (shared by Home, Score, Compare)
# ---------------------------------------------------------------------------

BORDERLINE = dict(GrAppv=100_000.0, SBA_Appv=75_000.0, Portion=0.75, Term=48, NoEmp=4, CreateJob=0, RetainedJob=4,
                  NewExist="Existing", IsFranchise="N", UrbanRural="Urban", RevLineCr="N", LowDoc="N",
                  Sector="Retail trade", State="CA", Bank="BANK OF AMERICA NATL ASSOC", BankState="CA",
                  DisbursementGross=100_000.0, Recession=True)
EXAMPLES = {
    "borderline": BORDERLINE,
    "low": {**BORDERLINE, "GrAppv": 500_000.0, "SBA_Appv": 375_000.0, "DisbursementGross": 500_000.0, "Term": 240,
            "NoEmp": 12, "RetainedJob": 12, "Sector": "Health care and social assistance"},
    "high": {**BORDERLINE, "GrAppv": 90_000.0, "SBA_Appv": 45_000.0, "Portion": 0.5, "DisbursementGross": 90_000.0,
             "Term": 36, "NewExist": "New", "RevLineCr": "Y", "BankState": "NC"},
}
EXAMPLE_LABELS = {"borderline": "Borderline", "low": "Low risk", "high": "High risk"}
YES_NO = {"Y": "Yes", "N": "No"}


def score_form_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Clean, engineer and score rows in the form layout."""
    clean, _ = cleaning.handle_outliers(form_to_clean(rows), params=load_cleaning_params())
    feats = build_model_frame(clean)
    return feats, predict(load_model(), feats)


@st.cache_data(show_spinner=False)
def score_examples(model_key: str) -> pd.DataFrame:
    """The three example applications with PD, band and decision."""
    rows = pd.DataFrame([{**v, "Recession": int(v["Recession"])} for v in EXAMPLES.values()], index=list(EXAMPLES))
    feats, p = score_form_rows(rows)
    out = rows.copy()
    out["p"] = p
    out["band"] = band_names(p)
    out["decision"] = decisions(p)
    out["Portion"] = feats["Portion"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Held-out portfolio with features (stress test, Monte Carlo, game, model lab)
# ---------------------------------------------------------------------------

def _clean_text(df: pd.DataFrame) -> pd.DataFrame:
    for c in features.CATEGORICAL_FEATURES + [features.BANK_COLUMN]:
        if c in df:
            df[c] = df[c].astype(str)
    return df


@st.cache_data(show_spinner="Scoring the committed test sample")
def sample_portfolio(model_key: str) -> pd.DataFrame:
    """The committed 20,000-loan stratified sample of the test split, rescored.

    Used by the pages that rescore or simulate many times (stress test, Monte
    Carlo, Beat the model), so they behave the same locally and on Streamlit
    Community Cloud.
    """
    df = _clean_text(pd.read_csv(TEST_SAMPLE_PATH))
    df["p"] = load_model().predict_proba(df[deployment_features()])[:, 1]
    return df


@st.cache_data(show_spinner="Scoring the held-out test portfolio with both models")
def model_lab_scores(model_key: str) -> tuple[pd.DataFrame, str]:
    """PD of the final and the comparison model on the held-out loans."""
    cols = list(dict.fromkeys(deployment_features() + ["Default"]))
    if config.FEATURES_DATA_PATH.exists() and TEST_IDS_PATH.exists():
        df = pd.read_parquet(config.FEATURES_DATA_PATH, columns=list(dict.fromkeys(cols + ["LoanNr_ChkDgt"])))
        ids = set(pd.read_parquet(TEST_IDS_PATH)["LoanNr_ChkDgt"])
        df = df[df["LoanNr_ChkDgt"].isin(ids)].reset_index(drop=True)
        source = f"full held-out test split ({len(df):,} loans, data/processed)"
    else:
        df = pd.read_csv(TEST_SAMPLE_PATH, usecols=lambda c: c in set(cols))
        source = f"stratified sample of the held-out test split ({len(df):,} loans, app/assets)"
    df = _clean_text(df)
    out = pd.DataFrame({"Default": df["Default"].astype(int).to_numpy()})
    out["p"] = load_model().predict_proba(df[deployment_features()])[:, 1]
    comp = load_comparison_model()
    if comp is not None:
        try:
            out["p_comp"] = comp.predict_proba(df[deployment_features()])[:, 1]
        except Exception:  # noqa: BLE001
            pass
    return out, source


# ---------------------------------------------------------------------------
# Committed aggregates (app/assets/aggregates, written by build_assets.py)
# ---------------------------------------------------------------------------

AGG_DIR = ASSETS_DIR / "aggregates"


@st.cache_data(show_spinner=False)
def _read_parquet(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path_str)


def load_aggregate(name: str) -> pd.DataFrame | None:
    path = AGG_DIR / name
    return _read_parquet(str(path), _mtime(path)) if path.exists() else None


def aggregate_manifest() -> dict:
    path = AGG_DIR / "manifest.json"
    return _load_json(str(path), _mtime(path)) if path.exists() else {}


@st.cache_resource(show_spinner=False)
def _reference(mtime: float):
    from analytics import prepare_reference
    ref = pd.read_parquet(AGG_DIR / "reference_sample.parquet")
    cols = [c for c in deployment_features() if c in ref.columns and pd.api.types.is_numeric_dtype(ref[c])]
    Z, mean, scale = prepare_reference(ref, cols)
    return ref, Z, mean, scale, cols


def reference_space():
    """Reference sample of training loans, standardised, for similar loans and percentiles (or None)."""
    path = AGG_DIR / "reference_sample.parquet"
    return _reference(_mtime(path)) if path.exists() else None


# ---------------------------------------------------------------------------
# State shared across pages: threshold, dark mode, drill-down filters
# ---------------------------------------------------------------------------

def shared_threshold() -> float:
    return float(st.session_state.get("shared_threshold", threshold()))


def set_shared_threshold(t: float) -> None:
    st.session_state["shared_threshold"] = float(t)


def threshold_slider(label: str = "Approval threshold (probability of default)", key: str = "thr",
                     lo: float = 0.01, hi: float = 0.99, step: float = 0.005, help_text: str | None = None) -> float:
    """Threshold slider that reads and writes the threshold shared by every page."""
    value = float(np.clip(round(shared_threshold() / step) * step, lo, hi))
    st.session_state[key] = value

    def sync():
        set_shared_threshold(st.session_state[key])

    t = st.slider(label, min_value=lo, max_value=hi, step=step, format="%.3f", key=key, on_change=sync,
                  help=help_text or "Loans with a probability of default below the threshold are approved. "
                                    "The value is shared with the other portfolio pages.")
    set_shared_threshold(t)
    return float(t)


US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def state_name(code: str) -> str:
    return US_STATES.get(str(code), str(code))


def filters() -> dict:
    """Drill-down filters shared by the explorer pages: {"state": code, "sector": name}."""
    return st.session_state.setdefault("filters", {})


def filter_version() -> int:
    return int(st.session_state.get("filter_version", 0))


def set_filter(name: str, value) -> None:
    f = filters()
    if value in (None, "", "All"):
        f.pop(name, None)
    else:
        f[name] = value


def clear_filters() -> None:
    st.session_state["filters"] = {}
    st.session_state["filter_version"] = filter_version() + 1  # remounts clickable charts without a selection


def apply_selection(event, chart_key: str, name: str, field: str = "location", lookup=None) -> bool:
    """Turn a new click on a chart into a filter. Returns True when the filter changed.

    The last selection seen for each chart is remembered, so a filter that
    was cleared is not set again by a stale selection.
    """
    points = []
    try:
        points = list(event.selection.points) if event is not None else []
    except AttributeError:
        points = list((event or {}).get("selection", {}).get("points", []))
    seen = st.session_state.setdefault("_seen_selection", {})
    signature = repr(points)
    if seen.get(chart_key) == signature:
        return False
    seen[chart_key] = signature
    if not points:
        return False
    pt = points[0]
    value = lookup(pt) if lookup is not None else pt.get(field)
    if value is None or filters().get(name) == value:
        return False
    set_filter(name, value)
    return True


def filter_chips(labels: dict | None = None, key: str = "clear_filters") -> None:
    """Chip row "Filtered: Florida, Retail trade" with a clear-all button."""
    f = filters()
    if not f:
        return
    names = {"state": lambda v: state_name(v), "sector": str, "bank": lambda v: str(v).title(),
             "pd_range": lambda v: f"PD {v[0]:.0%} to {v[1]:.0%}"}
    names.update(labels or {})
    chips = "".join(chip(names.get(k, str)(v), COLOURS["teal"]) for k, v in f.items())
    c1, c2 = st.columns([5, 1], vertical_alignment="center")
    c1.markdown(f'<div class="cl-filterbar"><span class="cl-filter-label">Filtered:</span>{chips}</div>',
                unsafe_allow_html=True)
    if c2.button("Clear all", key=key, icon=":material/filter_alt_off:", use_container_width=True):
        clear_filters()
        st.rerun()
