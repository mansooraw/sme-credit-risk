"""Cleaning functions for the SBA 7(a) loan data (CRISP-DM Data Preparation).

The module turns the raw Kaggle file SBAnational.csv into the analysis table
data/processed/loans_clean.parquet. Each step is a small function that takes
a DataFrame and returns a new one, so the steps can be shown one at a time in
notebook 02 and reused unchanged by the Streamlit app.

Main entry points
-----------------
load_raw(path)          read the raw CSV with every column as it is stored
clean_raw(df)           apply all cleaning steps in order and return the clean
                        table plus a report of what each step changed
quality_assessment(df)  data quality table (missing cells, invalid codes,
                        impossible values, duplicates) for raw or clean data

The decisions implemented here are explained, with evidence, in
notebooks/01_data_understanding.ipynb and notebooks/02_data_cleaning.ipynb.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

try:  # imported as part of the src package (notebooks, app)
    from src.config import DEFAULT_LABEL, PAID_LABEL, RAW_DATA_PATH, STATUS_COLUMN, TARGET
except ImportError:  # imported from inside the src folder
    from config import DEFAULT_LABEL, PAID_LABEL, RAW_DATA_PATH, STATUS_COLUMN, TARGET

# ---------------------------------------------------------------------------
# Constants used by the cleaning rules
# ---------------------------------------------------------------------------

CURRENCY_COLUMNS = ["DisbursementGross", "BalanceGross", "ChgOffPrinGr", "GrAppv", "SBA_Appv"]
DATE_COLUMNS = ["ApprovalDate", "DisbursementDate", "ChgOffDate"]
DATE_FORMAT = "%d-%b-%y"

# Loans in the file were approved between 1962 and 2014. Any parsed date
# outside this window cannot be right for this dataset.
VALID_DATE_MIN = pd.Timestamp("1960-01-01")
VALID_DATE_MAX = pd.Timestamp("2014-12-31")

# Text columns where a missing value becomes an explicit "Unknown" level
UNKNOWN = "Unknown"
FILL_UNKNOWN_COLUMNS = ["Name", "City", "Bank", "BankState"]

# Allowed values for the coded columns, raw coding and clean labels
VALID_CODES = {
    "RevLineCr": {"Y", "N", UNKNOWN},
    "LowDoc": {"Y", "N", UNKNOWN},
    "NewExist": {"1", "2", "Existing", "New", UNKNOWN},
    "UrbanRural": {"1", "2", "Urban", "Rural", UNKNOWN},
}

# RevLineCr recoding chosen on the evidence in notebook 01:
# "0" has the same profile as "N" and "T" has the same profile as "Y".
REVLINECR_MAP = {"Y": "Y", "N": "N", "0": "N", "T": "Y"}
LOWDOC_MAP = {"Y": "Y", "N": "N"}
NEWEXIST_MAP = {1: "Existing", 2: "New"}
URBANRURAL_MAP = {1: "Urban", 2: "Rural"}

# Near-duplicate key: the same borrower, place, approval date and amount
NEAR_DUPLICATE_KEY = ["Name", "City", "State", "ApprovalDate", "GrAppv"]

# Logical check settings
DISBURSEMENT_TOLERANCE_DAYS = 30
OUTLIER_QUANTILE = 0.999
NOEMP_PLACEHOLDER = 9999

# Loan size bands (approved amount in USD) used to impute a missing Term
LOAN_SIZE_BINS = [0, 50_000, 150_000, 500_000, 1_000_000, np.inf]
LOAN_SIZE_LABELS = ["<50k", "50k-150k", "150k-500k", "500k-1m", ">1m"]

# Minimum share of one state among known loans in a three-digit ZIP prefix
# before we accept that prefix as evidence for a missing State
ZIP_STATE_MIN_SHARE = 0.95
ZIP_STATE_MIN_LOANS = 20

CATEGORY_COLUMNS = [
    "City", "State", "Bank", "BankState", "RevLineCr", "RevLineCr_raw",
    "LowDoc", "NewExist", "UrbanRural", "MIS_Status",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path: str | Path = RAW_DATA_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw CSV exactly as stored.

    Only empty fields are treated as missing. The pandas default list of
    missing markers ("NA", "null" and so on) is switched off because "NA"
    could be a real abbreviation in a name or city.
    """
    return pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""], nrows=nrows)


# ---------------------------------------------------------------------------
# Step 1: parsing text into proper types
# ---------------------------------------------------------------------------

def parse_currency(s: pd.Series) -> pd.Series:
    """Convert text such as "$60,000.00 " to the float 60000.0.

    Dollar signs, thousands separators and spaces are removed. Anything that
    still cannot be read as a number becomes NaN.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype("string").str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def parse_sba_date(s: pd.Series, pivot_year: int | None = None) -> pd.Series:
    """Parse dates stored as "28-Feb-97" and fix the century.

    Python reads two-digit years 00-68 as 2000-2068 and 69-99 as 1969-1999,
    so an approval in "Dec-61" would become 2061. We apply our own rule: a
    two-digit year above the last two digits of the current year belongs to
    the 1900s, anything else to the 2000s. ``pivot_year`` can be passed to
    make the rule fixed (for example in tests).
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pivot_year is None:
        pivot_year = _dt.date.today().year % 100
    parsed = pd.to_datetime(s, format=DATE_FORMAT, errors="coerce")
    two_digit = parsed.dt.year % 100
    # Rebuild the full year from the two-digit year and the pivot
    full_year = np.where(two_digit > pivot_year, 1900 + two_digit, 2000 + two_digit)
    shift_years = full_year - parsed.dt.year
    needs_shift = parsed.notna() & (shift_years != 0)
    fixed = parsed.copy()
    fixed[needs_shift] = [
        d.replace(year=int(d.year + k)) for d, k in zip(parsed[needs_shift], shift_years[needs_shift])
    ]
    return fixed


def parse_fiscal_year(s: pd.Series) -> pd.Series:
    """Turn ApprovalFY text such as "1976A" into the integer 1976."""
    digits = s.astype("string").str.replace(r"\D", "", regex=True)
    return pd.to_numeric(digits, errors="coerce").astype("Int64")


def parse_types(df: pd.DataFrame, pivot_year: int | None = None) -> pd.DataFrame:
    """Apply the three parsers to every currency, date and fiscal year column."""
    out = df.copy()
    for col in CURRENCY_COLUMNS:
        out[col] = parse_currency(out[col])
    for col in DATE_COLUMNS:
        out[col] = parse_sba_date(out[col], pivot_year=pivot_year)
    out["ApprovalFY"] = parse_fiscal_year(out["ApprovalFY"])
    return out


# ---------------------------------------------------------------------------
# Step 2: target
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop loans with an unknown outcome and add the binary target.

    Default = 1 for "CHGOFF" (charged off) and 0 for "P I F" (paid in full).
    Rows with any other status, including missing, are removed because we
    cannot know whether they defaulted.
    """
    status = df[STATUS_COLUMN].astype("string").str.strip()
    keep = status.isin([DEFAULT_LABEL, PAID_LABEL]).fillna(False).to_numpy()
    out = df.loc[keep].copy()
    out[STATUS_COLUMN] = status[keep].astype(str)
    out[TARGET] = (out[STATUS_COLUMN] == DEFAULT_LABEL).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Step 3: near-duplicates
# ---------------------------------------------------------------------------

def remove_near_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Keep the first of several rows with the same borrower, place, date and amount.

    Every row has its own loan number, so there are no exact duplicates. Two
    records with the same Name, City, State, ApprovalDate and GrAppv are very
    likely the same loan entered twice, so only the first is kept.
    """
    subset = subset or NEAR_DUPLICATE_KEY
    return df.loc[~df.duplicated(subset=subset, keep="first")].copy()


# ---------------------------------------------------------------------------
# Step 4: coded categorical columns
# ---------------------------------------------------------------------------

def clean_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Recode RevLineCr, LowDoc, NewExist and UrbanRural to clean levels.

    Invalid or missing codes become an explicit "Unknown" level instead of
    being replaced by the most common value, because the unknown code may
    itself be related to default (tested in notebook 03). The original
    RevLineCr code is kept in RevLineCr_raw so that the recoding can be
    checked later.
    """
    out = df.copy()
    rev = out["RevLineCr"].astype("string").str.strip()
    out["RevLineCr_raw"] = rev.fillna("missing").astype(str)
    out["RevLineCr"] = rev.map(REVLINECR_MAP).fillna(UNKNOWN).astype(str)

    low = out["LowDoc"].astype("string").str.strip()
    out["LowDoc"] = low.map(LOWDOC_MAP).fillna(UNKNOWN).astype(str)

    new = pd.to_numeric(out["NewExist"], errors="coerce")
    out["NewExist"] = new.map(NEWEXIST_MAP).fillna(UNKNOWN).astype(str)

    urban = pd.to_numeric(out["UrbanRural"], errors="coerce")
    out["UrbanRural"] = urban.map(URBANRURAL_MAP).fillna(UNKNOWN).astype(str)
    return out


# ---------------------------------------------------------------------------
# Step 5: logical consistency checks
# ---------------------------------------------------------------------------

def apply_logical_checks(df: pd.DataFrame, tolerance_days: int = DISBURSEMENT_TOLERANCE_DAYS) -> pd.DataFrame:
    """Fix or flag values that contradict each other or cannot be true.

    * Dates outside 1960-2014 are impossible for this dataset and are set
      to missing (flag ``date_out_of_range``).
    * A disbursement more than ``tolerance_days`` before approval is kept but
      flagged with ``disb_before_approval``.
    * SBA_Appv above GrAppv is capped at GrAppv (the guarantee cannot exceed
      the loan).
    * Zip of 0 is not a real ZIP code and becomes missing.
    * A charge-off date on a paid loan, or no charge-off date on a charged-off
      loan, is flagged with ``chgoff_mismatch``. MIS_Status stays the label.
    """
    out = df.copy()

    out_of_range = pd.Series(False, index=out.index)
    for col in DATE_COLUMNS:
        bad = out[col].notna() & ((out[col] < VALID_DATE_MIN) | (out[col] > VALID_DATE_MAX))
        out_of_range |= bad
        out.loc[bad, col] = pd.NaT
    out["date_out_of_range"] = out_of_range

    lag_days = (out["DisbursementDate"] - out["ApprovalDate"]).dt.days
    out["disb_before_approval"] = (lag_days < -tolerance_days).fillna(False).astype(bool)

    out["SBA_Appv"] = np.minimum(out["SBA_Appv"], out["GrAppv"])

    zip_code = pd.to_numeric(out["Zip"], errors="coerce")
    out["Zip"] = zip_code.where(zip_code > 0).astype("Int64")

    has_chgoff_date = out["ChgOffDate"].notna()
    status = out[STATUS_COLUMN].astype(str)
    out["chgoff_mismatch"] = ((status == PAID_LABEL) & has_chgoff_date) | (
        (status == DEFAULT_LABEL) & ~has_chgoff_date
    )
    return out


# ---------------------------------------------------------------------------
# Step 6: outliers and impossible values
# ---------------------------------------------------------------------------

def naics_sector(naics: pd.Series) -> pd.Series:
    """Two-digit NAICS sector code as text, "00" when the code is unknown."""
    code = pd.to_numeric(naics, errors="coerce").fillna(0).astype(int)
    return code.astype(str).str.zfill(6).str[:2]


def loan_size_band(gr_appv: pd.Series) -> pd.Series:
    """Band of the approved loan amount, used to impute Term."""
    return pd.cut(gr_appv, bins=LOAN_SIZE_BINS, labels=LOAN_SIZE_LABELS, right=False).astype(str)


def handle_outliers(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Treat placeholder, impossible and extreme numeric values.

    * NoEmp of 9999 and values above the 99.9th percentile are set to
      missing and imputed with the median NoEmp of the same NAICS sector.
    * Term of 0 months is impossible for a loan; it is imputed with the
      median Term of loans in the same size band.
    * CreateJob and RetainedJob are capped (winsorised) at their 99.9th
      percentile.

    The thresholds and medians are returned in ``params``. Passing the same
    ``params`` again (for example in the app) applies the fitted values
    instead of recomputing them.
    """
    out = df.copy()
    fit = params is None
    params = {} if fit else params

    # NoEmp: placeholder 9999 and extreme head counts
    noemp = pd.to_numeric(out["NoEmp"], errors="coerce").astype(float)
    if fit:
        params["NoEmp_cap"] = float(noemp[noemp != NOEMP_PLACEHOLDER].quantile(OUTLIER_QUANTILE))
    bad_noemp = (noemp == NOEMP_PLACEHOLDER) | (noemp > params["NoEmp_cap"])
    noemp = noemp.mask(bad_noemp)
    sector = naics_sector(out["NAICS"])
    if fit:
        params["NoEmp_sector_median"] = noemp.groupby(sector).median().to_dict()
        params["NoEmp_overall_median"] = float(noemp.median())
    sector_median = sector.map(params["NoEmp_sector_median"]).fillna(params["NoEmp_overall_median"])
    out["noemp_imputed"] = noemp.isna()
    out["NoEmp"] = noemp.fillna(sector_median).round().astype(int)

    # Term: zero months is not a real loan term
    term = pd.to_numeric(out["Term"], errors="coerce").astype(float)
    term = term.mask(term <= 0)
    band = loan_size_band(out["GrAppv"])
    if fit:
        params["Term_band_median"] = term.groupby(band).median().to_dict()
        params["Term_overall_median"] = float(term.median())
    band_median = band.map(params["Term_band_median"]).fillna(params["Term_overall_median"])
    out["term_imputed"] = term.isna()
    out["Term"] = term.fillna(band_median).round().astype(int)

    # Job counts: winsorise the long right tail
    for col in ["CreateJob", "RetainedJob"]:
        values = pd.to_numeric(out[col], errors="coerce")
        key = f"{col}_cap"
        if fit:
            params[key] = float(values.quantile(OUTLIER_QUANTILE))
        out[col] = values.clip(upper=params[key]).round().astype(int)
    return out, params


# ---------------------------------------------------------------------------
# Step 7: missing text values
# ---------------------------------------------------------------------------

def zip_prefix_state_map(df: pd.DataFrame) -> dict:
    """Map three-digit ZIP prefixes to a state where the prefix is unambiguous.

    A prefix counts as unambiguous when at least ``ZIP_STATE_MIN_LOANS`` loans
    with a known state use it and one state accounts for at least
    ``ZIP_STATE_MIN_SHARE`` of them.
    """
    zip_code = pd.to_numeric(df["Zip"], errors="coerce")
    known = df["State"].notna() & (zip_code > 0)
    prefix = (zip_code[known] // 100).astype(int)
    table = pd.crosstab(prefix, df.loc[known, "State"])
    share = table.max(axis=1) / table.sum(axis=1)
    ok = (share >= ZIP_STATE_MIN_SHARE) & (table.sum(axis=1) >= ZIP_STATE_MIN_LOANS)
    return table.idxmax(axis=1)[ok].to_dict()


def fill_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing State from the ZIP prefix, other missing text with "Unknown".

    State is filled only when the ZIP prefix points to one state; otherwise
    it becomes "Unknown". Name, City, Bank and BankState become "Unknown".
    """
    out = df.copy()
    prefix_map = zip_prefix_state_map(out)
    missing_state = out["State"].isna()
    prefix = pd.to_numeric(out["Zip"], errors="coerce") // 100
    out["state_from_zip"] = False
    filled = prefix[missing_state].map(prefix_map)
    out.loc[missing_state, "State"] = filled
    out.loc[missing_state & out["State"].notna(), "state_from_zip"] = True
    out["State"] = out["State"].fillna(UNKNOWN)
    for col in FILL_UNKNOWN_COLUMNS:
        out[col] = out[col].fillna(UNKNOWN)
    return out


# ---------------------------------------------------------------------------
# Step 8: storage types
# ---------------------------------------------------------------------------

def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Use category and small integer types so the table is compact on disk."""
    out = df.copy()
    for col in CATEGORY_COLUMNS:
        if col in out:
            out[col] = out[col].astype("category")
    out["Name"] = out["Name"].astype(str)
    out["NAICS"] = pd.to_numeric(out["NAICS"], errors="coerce").astype("int32")
    out["FranchiseCode"] = pd.to_numeric(out["FranchiseCode"], errors="coerce").astype("int32")
    out["ApprovalFY"] = out["ApprovalFY"].astype("int16")
    for col in ["Term", "NoEmp", "CreateJob", "RetainedJob"]:
        out[col] = out[col].astype("int32")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def count_changed_cells(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """Number of cells whose value differs between two versions of a table.

    Only rows and columns present in both tables are compared, so removed
    rows and new flag columns are not counted here. Two missing values count
    as equal.
    """
    rows = before.index.intersection(after.index)
    total = 0
    for col in before.columns.intersection(after.columns):
        a = before.loc[rows, col]
        b = after.loc[rows, col]
        if a.dtype == b.dtype and a.equals(b):
            continue  # fast path: column untouched by this step
        a, b = a.astype(object), b.astype(object)
        same = (a == b) | (a.isna() & b.isna())
        total += int((~same).sum())
    return total


def clean_raw(df: pd.DataFrame, pivot_year: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Run every cleaning step in order.

    Order matters: rows are removed first (unknown label, near-duplicates) so
    that percentiles and medians used later are computed on the loans we
    actually keep.

    Returns
    -------
    df_clean : the cleaned table
    report   : dict with "steps" (DataFrame: rows before and after, rows
               removed, cells changed, new columns per step), "params"
               (outlier thresholds and imputation medians) and "flags"
               (number of rows flagged by each flag column)
    """
    steps = [
        ("parse_types", "Currency, dates (century fix) and ApprovalFY parsed", lambda d: parse_types(d, pivot_year)),
        ("add_target", "Loans with unknown MIS_Status dropped, Default added", add_target),
        ("remove_near_duplicates", "Near-duplicate loans removed", remove_near_duplicates),
        ("clean_codes", "RevLineCr, LowDoc, NewExist, UrbanRural recoded", clean_codes),
        ("apply_logical_checks", "Impossible dates, Zip 0, SBA_Appv cap, flags", apply_logical_checks),
        ("handle_outliers", "NoEmp, Term imputed; job counts winsorised", None),
        ("fill_missing_categoricals", "State from ZIP prefix; other text to Unknown", fill_missing_categoricals),
    ]
    records = []
    params: dict = {}
    current = df
    for name, description, func in steps:
        if name == "handle_outliers":
            result, params = handle_outliers(current)
        else:
            result = func(current)
        records.append({
            "step": name,
            "description": description,
            "rows_before": len(current),
            "rows_after": len(result),
            "rows_removed": len(current) - len(result),
            "cells_changed": count_changed_cells(current, result),
            "columns_added": ", ".join(c for c in result.columns if c not in current.columns),
        })
        current = result
    clean = optimise_dtypes(current)
    flag_cols = ["date_out_of_range", "disb_before_approval", "chgoff_mismatch",
                 "noemp_imputed", "term_imputed", "state_from_zip"]
    report = {
        "steps": pd.DataFrame(records),
        "params": params,
        "flags": {c: int(clean[c].sum()) for c in flag_cols},
    }
    return clean, report


# ---------------------------------------------------------------------------
# Data quality assessment
# ---------------------------------------------------------------------------

def _as_number(s: pd.Series) -> pd.Series:
    """Numeric view of a column whether it is still text or already parsed."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return parse_currency(s)


def _as_date(s: pd.Series) -> pd.Series:
    """Date view of a column whether it is still text or already parsed."""
    return s if pd.api.types.is_datetime64_any_dtype(s) else parse_sba_date(s)


def _code_text(s: pd.Series) -> pd.Series:
    """Non-missing values of a coded column as text ("1.0" becomes "1")."""
    values = s.dropna()
    if pd.api.types.is_numeric_dtype(values):
        values = values.astype(int)
    return values.astype(str).str.strip()


def quality_assessment(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Data quality table for the raw or the cleaned loan table.

    The same checks run on both versions so that the before and after
    numbers can be compared line by line. ``thresholds`` holds the 99.9th
    percentile caps from ``handle_outliers``; without it they are computed
    from ``df``.

    Returns a DataFrame with columns ``group``, ``check`` and ``value``.
    """
    rows = []

    def add(group: str, check: str, value) -> None:
        rows.append({"group": group, "check": check, "value": value})

    n_rows, n_cols = df.shape
    raw_cols = [c for c in df.columns if c in _RAW_COLUMNS]
    missing = int(df[raw_cols].isna().sum().sum())
    add("Size", "Rows", n_rows)
    add("Size", "Columns", n_cols)
    add("Missing", "Missing cells (original 27 columns)", missing)
    add("Missing", "Missing cells (% of original cells)", round(100 * missing / (n_rows * len(raw_cols)), 3))
    no_chgoff = [c for c in raw_cols if c != "ChgOffDate"]
    add("Missing", "Missing cells excluding ChgOffDate", int(df[no_chgoff].isna().sum().sum()))
    status = df[STATUS_COLUMN].astype("string").str.strip()
    add("Missing", "Rows with unknown MIS_Status", int((~status.isin([DEFAULT_LABEL, PAID_LABEL]).fillna(False)).sum()))

    text_typed = [c for c in CURRENCY_COLUMNS + DATE_COLUMNS + ["ApprovalFY"]
                  if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]))]
    add("Format", "Numeric or date columns stored as text", len(text_typed))

    for col, allowed in VALID_CODES.items():
        values = _code_text(df[col])
        add("Invalid codes", f"{col}: values outside the code list", int((~values.isin(allowed)).sum()))
    fy = df["ApprovalFY"]
    fy_bad = 0 if pd.api.types.is_numeric_dtype(fy) else int(fy.astype(str).str.contains(r"\D").sum())
    add("Invalid codes", "ApprovalFY: non-numeric values", fy_bad)

    noemp = _as_number(df["NoEmp"])
    term = _as_number(df["Term"])
    zip_code = _as_number(df["Zip"])
    gr = _as_number(df["GrAppv"])
    sba = _as_number(df["SBA_Appv"])
    disb = _as_number(df["DisbursementGross"])
    add("Impossible values", "NoEmp equal to placeholder 9999", int((noemp == NOEMP_PLACEHOLDER).sum()))
    add("Impossible values", "Term equal to 0 months", int((term == 0).sum()))
    add("Impossible values", "Zip equal to 0", int((zip_code == 0).sum()))
    add("Impossible values", "SBA_Appv above GrAppv", int((sba > gr).sum()))
    add("Impossible values", "DisbursementGross equal to 0", int((disb == 0).sum()))
    dates = {c: _as_date(df[c]) for c in DATE_COLUMNS}
    out_of_range = sum(int(((d < VALID_DATE_MIN) | (d > VALID_DATE_MAX)).sum()) for d in dates.values())
    add("Impossible values", "Dates outside 1960-2014", out_of_range)

    caps = thresholds or {}
    for col, series in [("NoEmp", noemp), ("CreateJob", _as_number(df["CreateJob"])),
                        ("RetainedJob", _as_number(df["RetainedJob"]))]:
        cap = caps.get(f"{col}_cap", float(series[series != NOEMP_PLACEHOLDER].quantile(OUTLIER_QUANTILE)))
        add("Extreme values", f"{col} above 99.9th percentile ({cap:,.0f})", int((series > cap).sum()))

    lag = (dates["DisbursementDate"] - dates["ApprovalDate"]).dt.days
    add("Logical", f"Disbursed over {DISBURSEMENT_TOLERANCE_DAYS} days before approval",
        int((lag < -DISBURSEMENT_TOLERANCE_DAYS).sum()))
    has_date = dates["ChgOffDate"].notna()
    add("Logical", "Paid in full but ChgOffDate present", int(((status == PAID_LABEL) & has_date).sum()))
    add("Logical", "Charged off but ChgOffDate missing", int(((status == DEFAULT_LABEL) & ~has_date).sum()))

    add("Duplicates", "Exact duplicate rows", int(df[raw_cols].duplicated().sum()))
    add("Duplicates", "Duplicate rows ignoring LoanNr_ChkDgt",
        int(df[[c for c in raw_cols if c != "LoanNr_ChkDgt"]].duplicated().sum()))
    key = df[NEAR_DUPLICATE_KEY].copy()
    key["ApprovalDate"] = dates["ApprovalDate"]
    key["GrAppv"] = gr
    add("Duplicates", "Near-duplicates (Name, City, State, ApprovalDate, GrAppv)", int(key.duplicated().sum()))
    return pd.DataFrame(rows)


def compare_quality(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Join two quality tables into one before/after table with the change."""
    table = before.rename(columns={"value": "before"}).merge(
        after[["group", "check", "value"]].rename(columns={"value": "after"}), on=["group", "check"], how="left",
    )
    table["change"] = table["after"] - table["before"]
    return table


# The 27 columns of the raw file, used so that flag columns added during
# cleaning do not distort the missing-cell and duplicate counts.
_RAW_COLUMNS = [
    "LoanNr_ChkDgt", "Name", "City", "State", "Zip", "Bank", "BankState", "NAICS",
    "ApprovalDate", "ApprovalFY", "Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob",
    "FranchiseCode", "UrbanRural", "RevLineCr", "LowDoc", "ChgOffDate", "DisbursementDate",
    "DisbursementGross", "BalanceGross", "MIS_Status", "ChgOffPrinGr", "GrAppv", "SBA_Appv",
]
