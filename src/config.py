"""Central configuration for the CreditLens project.

Every notebook, script and the Streamlit app imports paths and constants
from here so that nothing in the repository depends on a personal folder
layout. All paths are resolved relative to the repository root.
"""

from pathlib import Path

# Repository root: this file lives in <root>/src/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

RAW_DATA_PATH = RAW_DATA_DIR / "SBAnational.csv"
SAMPLE_DATA_PATH = SAMPLE_DATA_DIR / "sba_sample_5000.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "loans_clean.parquet"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "loans_features.parquet"

# Model artefacts
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

# Documentation and figures
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

# Kaggle dataset handle (public dataset, downloadable without an API key)
KAGGLE_DATASET = "mirbektoktogaraev/should-this-loan-be-approved-or-denied"
RAW_FILE_NAME = "SBAnational.csv"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Target definition
TARGET = "Default"
STATUS_COLUMN = "MIS_Status"
DEFAULT_LABEL = "CHGOFF"
PAID_LABEL = "P I F"

# Columns that are only known after the outcome (never used as features)
LEAKAGE_COLUMNS = ["ChgOffDate", "ChgOffPrinGr", "BalanceGross", "MIS_Status"]
IDENTIFIER_COLUMNS = ["LoanNr_ChkDgt", "Name"]

# Audit flags written by the cleaning step. They record what was repaired and
# are kept for traceability, but they must never be used as model features:
# for example 91% of loans with a recorded term of 0 defaulted, which means the
# zero was most likely written after the loan failed, so the flag would leak
# the outcome.
CLEANING_FLAG_COLUMNS = [
    "date_out_of_range",
    "disb_before_approval",
    "chgoff_mismatch",
    "noemp_imputed",
    "term_imputed",
    "state_from_zip",
]

# Great Recession window used for the Recession feature
RECESSION_START = "2007-12-01"
RECESSION_END = "2009-06-30"

# Business cost assumptions (shared by notebooks and the app)
NET_INTEREST_MARGIN = 0.05        # earned on a repaid loan, share of disbursed amount
LOSS_GIVEN_DEFAULT = 0.50         # share of the unguaranteed principal lost on default
WORKOUT_COST_RATE = 0.05          # administration and collection cost, share of disbursed amount
MIN_APPROVAL_RATE = 0.75          # business constraint on the decision threshold

# Shared plotting palette
PALETTE = {
    "navy": "#1F3A5F",
    "teal": "#2A9D8F",
    "amber": "#E9C46A",
    "red": "#E76F51",
    "grey": "#8D99AE",
}
PALETTE_LIST = list(PALETTE.values())

# NAICS two-digit sector names (31-33, 44-45 and 48-49 share a sector)
NAICS_SECTORS = {
    "11": "Agriculture, forestry, fishing and hunting",
    "21": "Mining, quarrying, and oil and gas extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale trade",
    "44": "Retail trade",
    "45": "Retail trade",
    "48": "Transportation and warehousing",
    "49": "Transportation and warehousing",
    "51": "Information",
    "52": "Finance and insurance",
    "53": "Real estate and rental and leasing",
    "54": "Professional, scientific, and technical services",
    "55": "Management of companies and enterprises",
    "56": "Administrative and support and waste management",
    "61": "Educational services",
    "62": "Health care and social assistance",
    "71": "Arts, entertainment, and recreation",
    "72": "Accommodation and food services",
    "81": "Other services (except public administration)",
    "92": "Public administration",
}
UNKNOWN_SECTOR = "Unknown"
