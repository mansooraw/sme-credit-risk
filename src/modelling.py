"""Shared modelling utilities for CreditLens (CRISP-DM Modeling).

This first version holds the pieces that every later notebook and the app
must share so that results are comparable:

get_feature_lists()                 numeric, categorical and bank feature names
split_data(df)                      the single stratified 80/20 train/test split;
                                    the test loan numbers are saved to
                                    data/processed/test_ids.parquet
load_test_ids()                     read the saved test loan numbers
build_preprocessor(numeric, categorical)
                                    ColumnTransformer used in front of every model

Added for notebooks 04 and 05:

build_tree_preprocessor(numeric, categorical)
                                    ColumnTransformer for the tree models
                                    (categories as integer codes, no one-hot)
make_pipeline(model, numeric, categorical, kind)
                                    preprocessing plus classifier in one Pipeline
stratified_subsample(X, y, n)       stratified row sample (RandomForest runs)
cv_folds(n_splits)                  the StratifiedKFold used for every comparison
cross_validate_model(pipe, X, y)    CV mean and std of ROC-AUC and PR-AUC

The cost based evaluation lives in src/evaluation.py.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:  # imported as part of the src package (notebooks, app)
    from src import config
    from src.features import BANK_COLUMN, BankEncoder, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
except ImportError:  # imported from inside the src folder
    import config
    from features import BANK_COLUMN, BankEncoder, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

ID_COLUMN = "LoanNr_ChkDgt"
TEST_IDS_PATH = config.PROCESSED_DATA_DIR / "test_ids.parquet"

# Categories seen fewer times than this in training are grouped into one
# "infrequent" column by the one-hot encoder (keeps rare BankState codes such
# as GU or VI from creating near-empty columns)
ONE_HOT_MIN_FREQUENCY = 100


def get_feature_lists() -> dict:
    """Feature names by role, as defined in src/features.py.

    Returns a dict with keys "numeric", "categorical", "bank" (the column
    handled by BankEncoder) and "all" (every column the pipeline receives).
    """
    return {
        "numeric": list(NUMERIC_FEATURES),
        "categorical": list(CATEGORICAL_FEATURES),
        "bank": BANK_COLUMN,
        "all": list(FEATURE_COLUMNS),
    }


def load_test_ids(path=TEST_IDS_PATH) -> pd.Series:
    """Loan numbers of the held-out test set saved by ``split_data``."""
    return pd.read_parquet(path)[ID_COLUMN]


def split_data(df: pd.DataFrame, feature_columns: list[str] | None = None, save_ids: bool = True,
               reuse_saved: bool = True):
    """Stratified 80/20 split used by every notebook and the app.

    The first call splits with ``train_test_split`` (stratified on Default,
    ``config.RANDOM_STATE``) and writes the test loan numbers to
    data/processed/test_ids.parquet. Later calls with ``reuse_saved=True``
    read that file and put exactly those loans in the test set, so the split
    stays identical even if a notebook filters or reorders the table.

    Parameters
    ----------
    df : table with LoanNr_ChkDgt, Default and the feature columns
    feature_columns : columns returned in X (default: FEATURE_COLUMNS)
    save_ids : write the test loan numbers when a new split is made
    reuse_saved : use the saved test loan numbers when the file exists

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    y = df[config.TARGET].astype(int)

    if reuse_saved and TEST_IDS_PATH.exists():
        test_mask = df[ID_COLUMN].isin(set(load_test_ids()))
    else:
        _, test_idx = train_test_split(
            df.index, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE,
        )
        test_mask = df.index.isin(test_idx)
        if save_ids:
            TEST_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.loc[test_mask, [ID_COLUMN]].reset_index(drop=True).to_parquet(TEST_IDS_PATH, index=False)

    test_mask = pd.Series(test_mask, index=df.index)
    X_train, X_test = df.loc[~test_mask, cols], df.loc[test_mask, cols]
    y_train, y_test = y[~test_mask], y[test_mask]
    return X_train, X_test, y_train, y_test


def build_preprocessor(numeric: list[str], categorical: list[str], bank_column: str | None = BANK_COLUMN,
                       min_frequency: int = ONE_HOT_MIN_FREQUENCY) -> ColumnTransformer:
    """ColumnTransformer placed in front of every model.

    * numeric columns: median imputation, then standard scaling
    * categorical columns: most-frequent imputation, then one-hot encoding
      with ``handle_unknown="ignore"`` (a level never seen in training
      becomes all zeros) and rare levels grouped by ``min_frequency``
    * bank column (if given): BankEncoder (BankSize and out-of-fold
      BankDefaultRate, fitted on training data only), then scaling

    Bank and City are never one-hot encoded; City is not a feature at all.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency)),
    ])
    transformers = [
        ("num", numeric_pipe, list(numeric)),
        ("cat", categorical_pipe, list(categorical)),
    ]
    if bank_column is not None:
        bank_pipe = Pipeline([
            ("encode", BankEncoder(column=bank_column)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("bank", bank_pipe, [bank_column]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# Additions for notebooks 04 and 05
# ---------------------------------------------------------------------------

def build_tree_preprocessor(numeric: list[str], categorical: list[str],
                            bank_column: str | None = BANK_COLUMN) -> ColumnTransformer:
    """ColumnTransformer placed in front of the tree models.

    Trees do not need scaling, and one-hot columns for State and BankState
    only make them slower, so this version keeps one column per feature:

    * numeric columns: median imputation (for incomplete app inputs)
    * categorical columns: integer codes from ``OrdinalEncoder``; a level
      never seen in training, or a missing value, becomes NaN, which the
      tree models route as missing
    * bank column (if given): BankEncoder (BankSize and out-of-fold
      BankDefaultRate, fitted on training data only)

    The output columns keep the feature names in the order numeric,
    categorical, bank, so SHAP values map one to one onto features.
    """
    numeric_pipe = SimpleImputer(strategy="median")
    categorical_pipe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                      encoded_missing_value=np.nan, dtype=np.float64)
    transformers = [
        ("num", numeric_pipe, list(numeric)),
        ("cat", categorical_pipe, list(categorical)),
    ]
    if bank_column is not None:
        transformers.append(("bank", BankEncoder(column=bank_column), [bank_column]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(model, numeric: list[str], categorical: list[str], kind: str = "tree",
                  bank_column: str | None = BANK_COLUMN) -> Pipeline:
    """Preprocessing plus classifier as one scikit-learn Pipeline.

    ``kind="linear"`` uses ``build_preprocessor`` (scaling and one-hot, for
    the logistic regression); ``kind="tree"`` uses ``build_tree_preprocessor``.
    The steps are named "prep" and "clf", so tuning grids use ``clf__<param>``.
    """
    if kind == "linear":
        prep = build_preprocessor(numeric, categorical, bank_column)
    elif kind == "tree":
        prep = build_tree_preprocessor(numeric, categorical, bank_column)
    else:
        raise ValueError("kind must be 'tree' or 'linear'")
    return Pipeline([("prep", prep), ("clf", model)])


def categorical_positions(numeric: list[str], categorical: list[str]) -> list[int]:
    """Positions of the categorical columns in the output of the tree preprocessor."""
    return list(range(len(numeric), len(numeric) + len(categorical)))


def stratified_subsample(X: pd.DataFrame, y: pd.Series, n: int, random_state: int = config.RANDOM_STATE):
    """Stratified sample of ``n`` rows, keeping the default rate of ``y``."""
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=random_state)
    return X_sub, y_sub


def cv_folds(n_splits: int = 5, random_state: int = config.RANDOM_STATE) -> StratifiedKFold:
    """The shuffled StratifiedKFold used for every model comparison."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_model(pipe, X, y, cv=None, n_jobs: int = 1, return_train_score: bool = False) -> dict:
    """Cross-validated ROC-AUC and PR-AUC with fit times.

    Returns a dict with the mean and standard deviation of each score, the
    mean fit time per fold and the total wall-clock time.
    """
    cv = cv if cv is not None else cv_folds()
    start = time.time()
    res = cross_validate(pipe, X, y, cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                         n_jobs=n_jobs, return_train_score=return_train_score)
    out = {
        "cv_roc_auc_mean": float(np.mean(res["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(res["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(res["test_pr_auc"])),
        "cv_pr_auc_std": float(np.std(res["test_pr_auc"])),
        "fit_seconds_per_fold": float(np.mean(res["fit_time"])),
        "cv_wall_seconds": float(time.time() - start),
    }
    if return_train_score:
        out["train_roc_auc_mean"] = float(np.mean(res["train_roc_auc"]))
        out["train_roc_auc_std"] = float(np.std(res["train_roc_auc"]))
    out["fold_roc_auc"] = [float(v) for v in res["test_roc_auc"]]
    return out
