"""Feature engineering for the SBA 7(a) loan data (CRISP-DM Data Preparation).

The module turns the cleaned loan table (data/processed/loans_clean.parquet)
into the model table. Every feature is defined in section 6 of the project
brief and in docs/data_dictionary.md, and is justified with evidence in
notebooks/03_eda_feature_engineering.ipynb.

Main entry points
-----------------
build_features(df)      add every row-level engineered feature to a cleaned
                        table (no information from other rows is used, so it
                        is safe to run before the train/test split and inside
                        the Streamlit app on a single loan)
BankEncoder             scikit-learn transformer that turns the lending bank
                        into BankSize (frequency encoding) and BankDefaultRate
                        (smoothed, out-of-fold target encoding). It needs the
                        labels, so it is fitted on training data only.
FEATURE_COLUMNS         columns passed to the model pipeline
DEPLOYMENT_FEATURES     FEATURE_COLUMNS without the calendar features; the
                        inputs of the deployed model (with
                        DEPLOYMENT_NUMERIC_FEATURES and
                        DEPLOYMENT_CATEGORICAL_FEATURES)
NUMERIC_FEATURES        numeric (including 0/1) model inputs
CATEGORICAL_FEATURES    low-cardinality categorical model inputs
BANK_COLUMN             the high-cardinality column handled by BankEncoder

Columns that are never features: the leakage columns, identifiers and audit
flags listed in src/config.py, plus City and Zip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

try:  # imported as part of the src package (notebooks, app)
    from src.config import (
        CLEANING_FLAG_COLUMNS, IDENTIFIER_COLUMNS, LEAKAGE_COLUMNS, NAICS_SECTORS,
        RANDOM_STATE, RECESSION_END, RECESSION_START, UNKNOWN_SECTOR,
    )
except ImportError:  # imported from inside the src folder
    from config import (
        CLEANING_FLAG_COLUMNS, IDENTIFIER_COLUMNS, LEAKAGE_COLUMNS, NAICS_SECTORS,
        RANDOM_STATE, RECESSION_END, RECESSION_START, UNKNOWN_SECTOR,
    )

UNKNOWN = "Unknown"

# Loans of 20 years or more are normally secured on real estate
REAL_ESTATE_TERM_MONTHS = 240

# Average days in a month, used to turn Term into an end date
DAYS_PER_MONTH = 365.25 / 12

# RevLineCr levels used as a feature. Notebook 03 shows that the raw code "T",
# which cleaning merged into "Y", defaults at 43% against 25% for "Y" and keeps
# that gap inside every approval year, so the feature keeps it as its own level.
REVLINECR_FEATURE_MAP = {"Y": "Y", "N": "N", "0": "N", "T": "T"}

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Engineered numeric features (section 6 of the brief)
ENGINEERED_NUMERIC = [
    "Portion", "TermYears", "LogDisbursement", "LoanPerEmployee", "JobsPerDollar",
    "DisbursedToApprovedRatio", "ApprovalYear", "ApprovalMonth", "DisbursementLagDays",
]
# Engineered 0/1 indicators, treated as numeric by the preprocessor
ENGINEERED_BINARY = [
    "RealEstate", "Recession", "NewBusiness", "IsFranchise", "SameStateBank",
    "DisbursementDateMissing",
]
# Cleaned numeric columns kept as they are
RAW_NUMERIC_KEPT = ["NoEmp", "CreateJob", "RetainedJob"]

NUMERIC_FEATURES = RAW_NUMERIC_KEPT + ENGINEERED_NUMERIC + ENGINEERED_BINARY
CATEGORICAL_FEATURES = ["Sector", "RevLineCr", "LowDoc", "UrbanRural", "NewExist", "State", "BankState"]
BANK_COLUMN = "Bank"
BANK_FEATURES = ["BankSize", "BankDefaultRate"]  # produced by BankEncoder inside the pipeline

# Everything the model pipeline receives. Bank is turned into BANK_FEATURES
# by BankEncoder; it is never one-hot encoded.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [BANK_COLUMN]

# Deployment feature set (decided in notebooks/04_model_selection.ipynb).
# ApprovalYear, ApprovalMonth and DisbursementLagDays describe when a loan was
# written and paid out. For a new application in 2026 they would lie outside
# the 1962-2014 training range, and in this data they partly measure how much
# of a loan's life had been observed when the file was extracted
# (right-censoring). They are therefore left out of the deployed model.
# Recession stays: the app uses it as a stress scenario switch.
CALENDAR_FEATURES = ["ApprovalYear", "ApprovalMonth", "DisbursementLagDays"]
# Term-based features dropped in the term sensitivity check of notebook 04
TERM_FEATURES = ["TermYears", "RealEstate"]

DEPLOYMENT_NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c not in CALENDAR_FEATURES]
DEPLOYMENT_CATEGORICAL_FEATURES = list(CATEGORICAL_FEATURES)
DEPLOYMENT_FEATURES = DEPLOYMENT_NUMERIC_FEATURES + DEPLOYMENT_CATEGORICAL_FEATURES + [BANK_COLUMN]

# Columns of the cleaned table used without engineering, for the ablation
# baseline in notebook 03 ("raw cleaned columns only")
RAW_BASELINE_NUMERIC = [
    "Term", "NoEmp", "CreateJob", "RetainedJob", "DisbursementGross", "GrAppv",
    "SBA_Appv", "NAICS", "FranchiseCode", "ApprovalFY",
]
RAW_BASELINE_CATEGORICAL = ["RevLineCr", "LowDoc", "UrbanRural", "NewExist", "State", "BankState"]

# Never allowed as features
EXCLUDED_COLUMNS = sorted(set(LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS + CLEANING_FLAG_COLUMNS
                              + ["City", "Zip", "RevLineCr_raw"]))


# ---------------------------------------------------------------------------
# Single-feature helpers
# ---------------------------------------------------------------------------

def naics_to_sector(naics: pd.Series) -> pd.Series:
    """Map NAICS codes to the 20 sector names; code 0 (unknown) becomes Unknown."""
    code = pd.to_numeric(naics, errors="coerce").fillna(0).astype(int)
    two_digit = code.astype(str).str.zfill(6).str[:2]
    return two_digit.map(NAICS_SECTORS).fillna(UNKNOWN_SECTOR)


def revlinecr_feature(df: pd.DataFrame) -> pd.Series:
    """RevLineCr with the raw code T kept as its own level.

    Uses RevLineCr_raw when present (training data). When only the cleaned
    RevLineCr is available (for example an app input of Y, N, T or Unknown)
    that value is used directly.
    """
    source = df["RevLineCr_raw"] if "RevLineCr_raw" in df else df["RevLineCr"]
    source = source.astype(str).str.strip()
    mapped = source.map(REVLINECR_FEATURE_MAP)
    return mapped.fillna(UNKNOWN)


def recession_active(start: pd.Series, term_months: pd.Series) -> pd.Series:
    """1 if a loan running from ``start`` for ``term_months`` overlaps the recession.

    The window is the NBER dating of the Great Recession in src/config.py
    (December 2007 to June 2009). The scheduled term is used, not the actual
    life of the loan, because the actual life is only known after the outcome.
    """
    end = start + pd.to_timedelta(term_months.astype(float) * DAYS_PER_MONTH, unit="D")
    overlap = (start <= pd.Timestamp(RECESSION_END)) & (end >= pd.Timestamp(RECESSION_START))
    return overlap.astype("int8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every row-level engineered feature to a cleaned loan table.

    Each feature uses only the loan's own row, so the function can run on the
    whole table before the split and on a single application in the app.
    The bank encodings need the labels and are therefore not built here; see
    ``BankEncoder``.

    Missing DisbursementDate (2,171 cleaned rows): the approval date is used
    as the start of the loan for Recession and the lag is therefore 0; the
    indicator DisbursementDateMissing records that the fallback was applied.

    Returns a copy of ``df`` with the new columns appended.
    """
    out = df.copy()
    gr = out["GrAppv"].astype(float)
    disb = out["DisbursementGross"].astype(float)
    term = out["Term"].astype(float)

    # Exposure and loan structure
    out["Portion"] = (out["SBA_Appv"].astype(float) / gr).clip(upper=1.0)
    out["RealEstate"] = (term >= REAL_ESTATE_TERM_MONTHS).astype("int8")
    out["TermYears"] = term / 12
    out["LogDisbursement"] = np.log1p(disb)
    out["DisbursedToApprovedRatio"] = disb / gr

    # Borrower profile
    new_exist = out["NewExist"].astype(str)
    out["NewBusiness"] = (new_exist == "New").astype("int8")
    out["IsFranchise"] = (~out["FranchiseCode"].isin([0, 1])).astype("int8")
    out["SameStateBank"] = (
        (out["BankState"].astype(str) == out["State"].astype(str)) & (out["State"].astype(str) != UNKNOWN)
    ).astype("int8")
    out["Sector"] = naics_to_sector(out["NAICS"])
    out["LoanPerEmployee"] = disb / np.maximum(out["NoEmp"].astype(float), 1)
    out["JobsPerDollar"] = (out["CreateJob"] + out["RetainedJob"]).astype(float) / disb * 1e5

    # Timing
    approval = pd.to_datetime(out["ApprovalDate"])
    disb_date = pd.to_datetime(out["DisbursementDate"])
    missing_disb = disb_date.isna()
    start = disb_date.fillna(approval)
    out["DisbursementDateMissing"] = missing_disb.astype("int8")
    out["ApprovalYear"] = approval.dt.year.astype("int16")
    out["ApprovalMonth"] = approval.dt.month.astype("int8")
    out["DisbursementLagDays"] = (start - approval).dt.days.astype("int32")
    out["Recession"] = recession_active(start, term)

    # Categorical inputs as plain text levels
    out["RevLineCr"] = revlinecr_feature(out)
    for col in ["Sector", "RevLineCr", "LowDoc", "UrbanRural", "NewExist", "State", "BankState", "Bank"]:
        out[col] = out[col].astype(str)
    return out


# ---------------------------------------------------------------------------
# Bank encodings
# ---------------------------------------------------------------------------

class BankEncoder(BaseEstimator, TransformerMixin):
    """Encode the lending bank as BankSize and BankDefaultRate.

    ``fit(X, y)`` learns, from training data only, the number of loans and
    the smoothed default rate of every bank:

        rate_bank = (defaults_bank + smoothing * global_rate) / (loans_bank + smoothing)

    so a bank with few loans is pulled towards the global default rate.
    ``transform`` maps each bank to these values; a bank never seen in
    training gets BankSize 0 and the global rate.

    ``fit_transform(X, y)`` is what a Pipeline calls on the training data. It
    returns out-of-fold rates: each training loan is encoded with statistics
    computed on the other ``n_splits - 1`` folds, so a loan's own label never
    enters its own feature. Without this, the model would learn from a
    feature that partly contains the answer and would look better in
    training than it is. Test data and app inputs always go through
    ``transform``, which uses statistics from the full training set only.

    Parameters
    ----------
    column : name of the bank column in X (X may also be a single column)
    smoothing : weight of the global rate, in loans
    n_splits : folds used for the out-of-fold training encoding
    random_state : seed for the fold assignment
    """

    def __init__(self, column: str = BANK_COLUMN, smoothing: float = 20.0, n_splits: int = 5,
                 random_state: int = RANDOM_STATE):
        self.column = column
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    # helpers --------------------------------------------------------------
    def _bank(self, X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            s = X[self.column] if self.column in X else X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(np.asarray(X).reshape(len(X), -1)[:, 0])
        return s.astype(str).reset_index(drop=True)

    def _stats(self, bank: pd.Series, y: np.ndarray) -> tuple[pd.Series, pd.Series, float]:
        prior = float(np.mean(y))
        grouped = pd.DataFrame({"bank": bank.to_numpy(), "y": y}).groupby("bank")["y"].agg(["sum", "count"])
        rate = (grouped["sum"] + self.smoothing * prior) / (grouped["count"] + self.smoothing)
        return grouped["count"], rate, prior

    # scikit-learn API -----------------------------------------------------
    def fit(self, X, y):
        bank = self._bank(X)
        y = np.asarray(y, dtype=float)
        self.size_map_, self.rate_map_, self.global_rate_ = self._stats(bank, y)
        self.n_features_in_ = 1
        return self

    def transform(self, X) -> np.ndarray:
        bank = self._bank(X)
        size = bank.map(self.size_map_).fillna(0).to_numpy(dtype=float)
        rate = bank.map(self.rate_map_).fillna(self.global_rate_).to_numpy(dtype=float)
        return np.column_stack([size, rate])

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        if y is None:
            raise ValueError("BankEncoder needs the target y to fit")
        self.fit(X, y)
        bank = self._bank(X)
        y = np.asarray(y, dtype=float)
        size = bank.map(self.size_map_).to_numpy(dtype=float)
        rate = np.empty(len(bank), dtype=float)
        folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fit_idx, enc_idx in folds.split(np.zeros(len(y)), y):
            _, fold_rate, fold_prior = self._stats(bank.iloc[fit_idx], y[fit_idx])
            rate[enc_idx] = bank.iloc[enc_idx].map(fold_rate).fillna(fold_prior).to_numpy(dtype=float)
        return np.column_stack([size, rate])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(BANK_FEATURES, dtype=object)


# ---------------------------------------------------------------------------
# Labels and help text (used by models/feature_schema.json and the app form)
# ---------------------------------------------------------------------------

FEATURE_INFO = {
    "NoEmp": ("Number of employees", "Head count of the business at application."),
    "CreateJob": ("Jobs to be created", "New jobs the business expects the loan to create."),
    "RetainedJob": ("Jobs retained", "Existing jobs the loan is expected to keep."),
    "Portion": ("Guaranteed share", "Share of the loan covered by the government guarantee (0 to 1)."),
    "TermYears": ("Term (years)", "Scheduled loan term in years."),
    "LogDisbursement": ("Log of disbursed amount", "Natural log of (1 + amount disbursed in USD)."),
    "LoanPerEmployee": ("Loan per employee (USD)", "Disbursed amount divided by the number of employees (at least 1)."),
    "JobsPerDollar": ("Jobs per USD 100,000", "Created plus retained jobs per USD 100,000 disbursed."),
    "DisbursedToApprovedRatio": ("Disbursed to approved ratio", "Amount disbursed divided by the gross amount approved."),
    "ApprovalYear": ("Approval year", "Calendar year in which the guarantee was approved."),
    "ApprovalMonth": ("Approval month", "Calendar month of approval (1 to 12)."),
    "DisbursementLagDays": ("Days from approval to disbursement", "Days between approval and disbursement; 0 when the disbursement date is not known."),
    "RealEstate": ("Real-estate term (20 years or more)", "1 if the term is 240 months or more, which normally means the loan is secured on property."),
    "Recession": ("Active during the 2007-2009 recession", "1 if the scheduled life of the loan overlaps December 2007 to June 2009."),
    "NewBusiness": ("New business", "1 if the business is two years old or less."),
    "IsFranchise": ("Franchise", "1 if the business operates under a franchise agreement."),
    "SameStateBank": ("Lender in the same state", "1 if the lending bank is in the borrower's state."),
    "DisbursementDateMissing": ("Disbursement date unknown", "1 if no disbursement date was recorded and the approval date was used instead."),
    "Sector": ("Industry sector", "NAICS sector of the business; Unknown when no industry code was recorded."),
    "RevLineCr": ("Revolving line of credit", "Y, N, T (an undocumented code in the source data that behaves differently from Y) or Unknown."),
    "LowDoc": ("LowDoc programme", "Y if the loan used the one-page low-documentation application."),
    "UrbanRural": ("Urban or rural", "Location type of the business; Unknown for loans where it was not recorded."),
    "NewExist": ("Business age group", "Existing (more than two years), New (two years or less) or Unknown."),
    "State": ("Borrower state", "Two-letter US state code of the business."),
    "BankState": ("Bank state", "Two-letter state code of the lending bank."),
    "Bank": ("Lending bank", "Name of the lending bank; encoded as BankSize and BankDefaultRate inside the model."),
    "BankSize": ("Bank size (loans in training data)", "Number of training loans made by the same bank; 0 for a bank not seen in training."),
    "BankDefaultRate": ("Bank historical default rate", "Smoothed default rate of the bank's training loans; the overall rate for a new bank."),
}
